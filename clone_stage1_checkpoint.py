#!/usr/bin/env python3
"""Clone a completed stage-1 checkpoint to a new run id.

Usage:
  python3 clone_stage1_checkpoint.py \
    --output-dir out/out \
    --source-run-id full_20260413_203931 \
    --new-run-id full_20260413_203931_softskills_v1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path


def _checkpoint_path(output_dir: Path, run_id: str) -> Path:
    return output_dir / "checkpoints" / f"{run_id}_stage1_extracted.jsonl"


def clone_stage1_checkpoint(
    output_dir: Path,
    source_run_id: str,
    new_run_id: str,
    overwrite: bool = False,
) -> Path:
    src = _checkpoint_path(output_dir, source_run_id)
    dst = _checkpoint_path(output_dir, new_run_id)

    if not src.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src}")
    if src.resolve() == dst.resolve():
        raise ValueError("Source and destination run ids must differ")
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"Destination checkpoint already exists: {dst}\n"
            "Use --overwrite to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cloned_at = dt.datetime.now(dt.timezone.utc).isoformat()
    wrote_header = False
    data_records = 0

    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_num, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                obj["run_id"] = new_run_id
                obj["cloned_from_run_id"] = source_run_id
                obj["cloned_at"] = cloned_at
                wrote_header = True
            elif obj.get("_complete"):
                obj["cloned_from_run_id"] = source_run_id
                obj["cloned_at"] = cloned_at
            else:
                data_records += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if not wrote_header:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"Checkpoint header missing in {src}")
    if data_records == 0:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"No stage-1 records found in {src}")

    return dst


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clone a completed stage-1 checkpoint to a new run id."
    )
    parser.add_argument(
        "--output-dir",
        default="out/out",
        help="Pipeline output directory that contains the checkpoints/ folder (default: out/out)",
    )
    parser.add_argument("--source-run-id", required=True, help="Existing run id to clone from")
    parser.add_argument("--new-run-id", required=True, help="New run id for the cloned stage-1 checkpoint")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination stage-1 checkpoint if it already exists",
    )
    args = parser.parse_args()

    try:
        dst = clone_stage1_checkpoint(
            output_dir=Path(args.output_dir),
            source_run_id=args.source_run_id.strip(),
            new_run_id=args.new_run_id.strip(),
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Cloned stage-1 checkpoint to: {dst}")
    print("Next run:")
    print(
        "  python -m skills_extraction "
        f"--output-dir {args.output_dir} "
        f"--run-id {args.new_run_id}"
    )
    print("To retry old stage-1 failures before stages 2-4:")
    print(
        "  python -m skills_extraction "
        f"--output-dir {args.output_dir} "
        f"--run-id {args.new_run_id} "
        "--retry-stage1-errors"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
