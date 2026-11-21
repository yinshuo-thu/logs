"""Linux cgroup-v2 job launcher; atomic bounded-memory EXPORT writes."""
import argparse
import contextlib
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

HARD = 0.85
STOP = 0.88
CHUNK = 4 * 1024**2


def cgroups():
    """Read the unified hierarchy as exposed to this process, including ancestors."""
    rel = next(x[3:] for x in Path("/proc/self/cgroup").read_text().splitlines() if x.startswith("0::"))
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        if not right.startswith("cgroup2 "):
            continue
        root, mount = [Path(s.replace(r"\040", " ")) for s in left.split()[3:5]]
        relpath = Path(rel)
        for candidate in [mount / relpath.relative_to(root) if relpath.is_relative_to(root) else mount / rel.lstrip("/"),
                          mount / rel.lstrip("/")]:
            if not candidate.is_relative_to(mount) or ".." in candidate.parts:
                continue
            if (candidate / "cgroup.procs").exists():
                return [candidate, *(p for p in candidate.parents if p.is_relative_to(mount))]
    raise RuntimeError("cannot resolve cgroup v2; use a dedicated Linux job allocation")


def memory_levels():
    info = {s.split(':')[0]: int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()
            if s.startswith(('MemTotal:', 'MemAvailable:'))}
    levels = [(info['MemTotal'], info['MemTotal'] - info['MemAvailable'])]
    for p in cgroups():
        f = p / "memory.max"
        if f.exists() and (v := f.read_text().strip()) != "max":
            levels.append((int(v), int((p / "memory.current").read_text())))
    return levels


def disk(path):
    stat = os.statvfs(path)
    # Match df's non-root view: reserved blocks cannot be spent by this writer.
    used = (stat.f_blocks - stat.f_bfree) * stat.f_frsize
    return used + stat.f_bavail * stat.f_frsize, used


def headroom(total, used, threshold):
    return int(total * threshold) - used


class StreamWriter:
    def __init__(self, file, dirname):
        self.file = file
        self.dirname = dirname

    def tell(self):
        return self.file.tell()

    def flush(self):
        self.file.flush()

    def write(self, data):
        buf = memoryview(data).cast("B")
        for i in range(0, len(buf), CHUNK):
            part = buf[i:i+CHUNK]
            total, used = disk(self.dirname)
            if len(part) > headroom(total, used, STOP):
                raise OSError("EXPORT filesystem would exceed 88%; previous output preserved")
            if self.file.write(part) != len(part):
                raise OSError("short write")
        return len(buf)


@contextlib.contextmanager
def atomic_export(root, relative_name):
    """Supply the same EXPORT root to all writers; temp and target share a filesystem."""
    root = Path(root).resolve()
    dest = (root / relative_name).resolve()
    if not dest.is_relative_to(root) or dest == root or dest.name == ".export.lock":
        raise ValueError("output must be under EXPORT")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (root / ".export.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fd, name = tempfile.mkstemp(dir=dest.parent, prefix=".partial-")
        try:
            with os.fdopen(fd, "wb", buffering=0) as file:
                yield StreamWriter(file, dest.parent)
                os.fsync(file.fileno())
            os.replace(name, dest)
        finally:
            Path(name).unlink(missing_ok=True)


def terminate(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
        time.sleep(1)
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def supervise(root, cap, command):
    leaf = cgroups()[0]
    limit_file = leaf / "memory.max"
    value = limit_file.read_text().strip() if limit_file.exists() else "max"
    if value == "max" or int(value) > cap:
        raise RuntimeError("kernel RAM cap missing or too large; refusing to launch workload")
    if headroom(*disk(root), STOP) <= 0:
        raise RuntimeError("EXPORT filesystem already at stop threshold")
    started = time.monotonic()
    peaks = [0.0, 0.0, 0.0]
    next_log = started
    child = subprocess.Popen(command, start_new_session=True)
    try:
        while child.poll() is None:
            levels = memory_levels()
            # The immediate leaf is already capped; use its high watermark for reclaim,
            # not a second fractional job cap. Monitor host and parent limits separately.
            host, parents = levels[0], levels[2:]
            fs = disk(root)
            readings = [host[1] / host[0], levels[1][1] / levels[1][0], fs[1] / fs[0]]
            peaks = [max(p, r) for p, r in zip(peaks, readings)]
            if time.monotonic() >= next_log:
                print(f"peak ratios host_RAM/job_cap/EXPORT_fs: {peaks}", flush=True)
                next_log = time.monotonic() + 60
            if (any(headroom(t, u, STOP) <= 0 for t, u in [host, *parents])
                    or headroom(*fs, STOP) <= 0 or time.monotonic() - started >= 36*3600):
                print("budget stop: host/parent RAM, EXPORT space, or 36h deadline", file=sys.stderr)
                return 75
            time.sleep(1)
        return child.returncode
    finally:
        if child.poll() is None:
            terminate(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--existing-scope", action="store_true",
                        help="use a dedicated scheduler cgroup with an already enforced memory.max")
    parser.add_argument("--verify-cap", type=int, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply -- followed by the whole experiment command")
    root = args.export.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["EXPORT_DIR"] = str(root)
    if args.verify_cap is not None:
        return supervise(root, args.verify_cap, command)
    levels = memory_levels()
    if args.existing_scope:
        leaf = cgroups()[0]
        if len(levels) < 2 or not (leaf / "memory.max").is_file() or (leaf / "memory.max").read_text().strip() == "max":
            raise RuntimeError("existing job needs a dedicated, finite leaf memory.max")
        # Current job usage is already counted by its enclosing domains.
        cap = levels[1][1] + min(headroom(t, u, HARD) for t, u in [levels[0], *levels[2:]])
        return supervise(root, cap, command)
    cap = min(headroom(t, u, HARD) for t, u in levels)
    if cap <= 0 or headroom(*disk(root), STOP) <= 0:
        raise RuntimeError("insufficient RAM or EXPORT space before launch")
    high = max(1, int(cap * .94))
    print(f"whole-job MemoryHigh={high} MemoryMax={cap} EXPORT={root}", flush=True)
    cmd = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryHigh={high}",
           "-p", f"MemoryMax={cap}", "-p", "MemorySwapMax=0", sys.executable,
           str(Path(__file__).resolve()), "--export", str(root), "--verify-cap", str(cap), "--", *command]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
