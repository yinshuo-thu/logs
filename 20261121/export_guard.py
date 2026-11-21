"""Point-in-time resource reading for the agent after a training or export stage."""
import argparse
import json
import shutil
from pathlib import Path


def resource_snapshot(export_dir):
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            meminfo[key] = int(value.split()[0])
    ram = 100 * (1 - meminfo["MemAvailable"] / meminfo["MemTotal"])
    disk = shutil.disk_usage(export_dir)
    used_disk = 100 * disk.used / (disk.used + disk.free)
    return {"host_ram_pct": round(ram, 2), "export_filesystem_pct": round(used_disk, 2),
            "review_before_next_chunk": max(ram, used_disk) >= 88}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path, help="existing EXPORT directory")
    args = parser.parse_args()
    print(json.dumps(resource_snapshot(args.export)))
