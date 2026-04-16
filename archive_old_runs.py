#!/usr/bin/env python3
"""Archive older SkillsExtraction run artifacts out of the output directory.

By default this script performs a dry run and shows which run IDs would be
archived. Use ``--apply`` to actually move files.

Default behavior:
- scan the entire output directory recursively
- group artifacts by run id
- keep the N most recent run ids
- move older run artifacts to a sibling archive directory outside the repo
- move loose log/data files that are not tied to a run into the archive batch
- delete only explicit transient junk such as ``*.tmp`` and ``nohup.out``
- write an ``archive_manifest.json`` and prune empty directories afterward

Example:
  python3 archive_old_runs.py
  python3 archive_old_runs.py --apply
  python3 archive_old_runs.py --apply --keep-latest 3
  python3 archive_old_runs.py --apply --run-id 20260402_180039 --run-id 20260402_182108
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "out" / "out"
DEFAULT_ARCHIVE_ROOT = REPO_ROOT.parent / "skills_extraction_archive"

_RUN_FILE_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(
        r"^SkillsExtraction_"
        r"(?:augmented|low_confidence|mentions|pipeline|quality|skill_frequency|job_skills)"
        r"_run_(.+)\.(?:json|jsonl|csv|log)$"
    ),
    re.compile(r"^SkillsExtraction_run_summary_(.+)\.json$"),
)
_CHECKPOINT_PATTERN = re.compile(r"^(.+?)_stage\d+_[^.]+\.jsonl$")
_RUN_ID_TIMESTAMP_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"^full_(\d{8}_\d{6})$"),
    re.compile(r"^(\d{8}_\d{6})$"),
)
_TRANSIENT_NAME_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"^\.(?:DS_Store|_.*)$"),
    re.compile(r"^nohup\.out$"),
    re.compile(r".*\.tmp$"),
    re.compile(r".*\.partial$"),
    re.compile(r".*\.part$"),
    re.compile(r".*\.lock$"),
    re.compile(r".*\.pid$"),
    re.compile(r".*\.bak$"),
    re.compile(r".*\.sw[op]$"),
)


def _parse_run_id(name: str) -> Optional[str]:
    for pattern in _RUN_FILE_PATTERNS:
        match = pattern.match(name)
        if match:
            return match.group(1)
    return None


def _parse_checkpoint_run_id(name: str) -> Optional[str]:
    match = _CHECKPOINT_PATTERN.match(name)
    if match:
        return match.group(1)
    return None


def _parse_run_timestamp(run_id: str) -> Optional[dt.datetime]:
    for pattern in _RUN_ID_TIMESTAMP_PATTERNS:
        match = pattern.match(run_id)
        if match:
            return dt.datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    return None


def _is_transient_file(path: Path) -> bool:
    return any(pattern.match(path.name) for pattern in _TRANSIENT_NAME_PATTERNS)


def _iter_output_files(output_dir: Path) -> Iterable[Path]:
    for child in sorted(output_dir.rglob("*")):
        if child.is_file():
            yield child


def _collect_inventory(
    output_dir: Path,
) -> Tuple[Dict[str, List[Path]], List[Path], List[Path]]:
    files_by_run: DefaultDict[str, List[Path]] = defaultdict(list)
    loose_archive_files: List[Path] = []
    transient_delete_files: List[Path] = []

    for child in _iter_output_files(output_dir):
        run_id = _parse_run_id(child.name)
        if run_id is None:
            run_id = _parse_checkpoint_run_id(child.name)

        if run_id:
            files_by_run[run_id].append(child)
        elif _is_transient_file(child):
            transient_delete_files.append(child)
        else:
            loose_archive_files.append(child)

    return dict(files_by_run), loose_archive_files, transient_delete_files


def _latest_mtime(paths: Iterable[Path]) -> float:
    return max(path.stat().st_mtime for path in paths)


def _run_sort_key(run_id: str, paths: Sequence[Path]) -> Tuple[float, str]:
    parsed_timestamp = _parse_run_timestamp(run_id)
    if parsed_timestamp is not None:
        return parsed_timestamp.timestamp(), run_id
    return _latest_mtime(paths), run_id


def _select_runs_to_archive(
    files_by_run: Dict[str, List[Path]],
    keep_latest: int,
    keep_run_ids: Sequence[str],
    explicit_run_ids: Sequence[str],
) -> Tuple[List[str], List[str]]:
    known_run_ids = set(files_by_run)
    requested_run_ids = [rid for rid in explicit_run_ids if rid in known_run_ids]
    missing_run_ids = sorted(set(explicit_run_ids) - known_run_ids)

    if requested_run_ids:
        selected = sorted(set(requested_run_ids))
        return selected, missing_run_ids

    ranked = sorted(
        files_by_run.items(),
        key=lambda item: _run_sort_key(item[0], item[1]),
        reverse=True,
    )
    keep = set(keep_run_ids)
    if keep_latest > 0:
        keep.update(run_id for run_id, _paths in ranked[:keep_latest])

    selected = [run_id for run_id, _paths in ranked if run_id not in keep]
    return selected, missing_run_ids


def _describe_plan(run_ids: Sequence[str], files_by_run: Dict[str, List[Path]]) -> str:
    lines: List[str] = []
    total_files = 0
    total_bytes = 0
    for run_id in run_ids:
        paths = sorted(files_by_run[run_id])
        file_count = len(paths)
        byte_count = sum(path.stat().st_size for path in paths)
        total_files += file_count
        total_bytes += byte_count
        lines.append(
            f"- {run_id}: {file_count} files, {byte_count / (1024 ** 3):.2f} GiB"
        )
    lines.append(
        f"Total: {len(run_ids)} run(s), {total_files} files, {total_bytes / (1024 ** 3):.2f} GiB"
    )
    return "\n".join(lines)


def _move_run_files(
    run_id: str,
    paths: Sequence[Path],
    output_dir: Path,
    archive_batch_dir: Path,
) -> None:
    run_archive_dir = archive_batch_dir / run_id
    for src in sorted(paths):
        relative_parent = src.relative_to(output_dir).parent
        dst_dir = run_archive_dir / relative_parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists():
            raise FileExistsError(f"Archive destination already exists: {dst}")
        shutil.move(str(src), str(dst))


def _move_loose_files(
    paths: Sequence[Path],
    output_dir: Path,
    archive_batch_dir: Path,
) -> None:
    loose_root = archive_batch_dir / "_loose_files"
    for src in sorted(paths):
        relative_parent = src.relative_to(output_dir).parent
        dst_dir = loose_root / relative_parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists():
            raise FileExistsError(f"Archive destination already exists: {dst}")
        shutil.move(str(src), str(dst))


def _delete_transient_files(paths: Sequence[Path]) -> None:
    for src in sorted(paths):
        src.unlink()


def _prune_empty_dirs(root_dir: Path) -> None:
    for child in sorted(root_dir.rglob("*"), reverse=True):
        if child.is_dir():
            try:
                child.rmdir()
            except OSError:
                continue


def _bytes_for_paths(paths: Sequence[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def _build_run_manifest_entries(
    output_dir: Path,
    run_ids: Sequence[str],
    files_by_run: Dict[str, List[Path]],
) -> List[Dict[str, Any]]:
    return [
        {
            "run_id": run_id,
            "file_count": len(files_by_run[run_id]),
            "bytes": _bytes_for_paths(files_by_run[run_id]),
            "files": [
                str(path.relative_to(output_dir))
                for path in sorted(files_by_run[run_id])
            ],
        }
        for run_id in run_ids
    ]


def _write_manifest(
    manifest_path: Path,
    archive_root: Path,
    kept_run_ids: Sequence[str],
    source_output_dir: Path,
    archived_run_entries: Sequence[Dict[str, Any]],
    archived_loose_files: Sequence[str],
    deleted_transient_files: Sequence[str],
) -> None:
    manifest: Dict[str, Any] = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_output_dir": str(source_output_dir),
        "archive_root": str(archive_root),
        "archived_runs": list(archived_run_entries),
        "kept_run_ids": list(kept_run_ids),
        "archived_loose_files": list(archived_loose_files),
        "deleted_transient_files": list(deleted_transient_files),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def archive_runs(
    output_dir: Path,
    archive_root: Path,
    keep_latest: int,
    keep_run_ids: Sequence[str],
    explicit_run_ids: Sequence[str],
    apply: bool,
) -> int:
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    files_by_run, loose_archive_files, transient_delete_files = _collect_inventory(
        output_dir
    )
    if not files_by_run and not loose_archive_files and not transient_delete_files:
        print(f"No run artifacts found under {output_dir}")
        return 0

    selected_run_ids, missing_run_ids = _select_runs_to_archive(
        files_by_run,
        keep_latest=keep_latest,
        keep_run_ids=keep_run_ids,
        explicit_run_ids=explicit_run_ids,
    )

    if missing_run_ids:
        print("Requested run ids not found:")
        for run_id in missing_run_ids:
            print(f"- {run_id}")

    ranked = sorted(
        files_by_run.items(),
        key=lambda item: _run_sort_key(item[0], item[1]),
        reverse=True,
    )
    kept_run_ids = [
        run_id
        for run_id, _paths in ranked
        if run_id not in set(selected_run_ids)
    ]

    if not selected_run_ids and not loose_archive_files and not transient_delete_files:
        print("Nothing selected for archiving.")
        return 0

    print(f"Output dir:   {output_dir}")
    print(f"Archive root: {archive_root}")
    if kept_run_ids:
        print("Keeping in place:")
        for run_id in kept_run_ids:
            print(f"- {run_id}")
    print("Plan:")
    if selected_run_ids:
        print(_describe_plan(selected_run_ids, files_by_run))
    else:
        print("- No run-bound artifacts selected.")
    if loose_archive_files:
        print(
            f"Loose files to archive: {len(loose_archive_files)} files, "
            f"{_bytes_for_paths(loose_archive_files) / (1024 ** 3):.2f} GiB"
        )
    if transient_delete_files:
        print(f"Transient files to delete: {len(transient_delete_files)}")

    if not apply:
        print("\nDry run only. Re-run with --apply to move these files.")
        return 0

    archived_run_entries = _build_run_manifest_entries(
        output_dir=output_dir,
        run_ids=selected_run_ids,
        files_by_run=files_by_run,
    )
    archived_loose_file_paths = [
        str(path.relative_to(output_dir)) for path in sorted(loose_archive_files)
    ]
    deleted_transient_file_paths = [
        str(path.relative_to(output_dir)) for path in sorted(transient_delete_files)
    ]

    archive_root.mkdir(parents=True, exist_ok=True)
    batch_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_batch_dir = archive_root / f"archive_{batch_id}"
    archive_batch_dir.mkdir(parents=True, exist_ok=False)

    for run_id in selected_run_ids:
        _move_run_files(run_id, files_by_run[run_id], output_dir, archive_batch_dir)
    if loose_archive_files:
        _move_loose_files(loose_archive_files, output_dir, archive_batch_dir)
    if transient_delete_files:
        _delete_transient_files(transient_delete_files)

    _write_manifest(
        manifest_path=archive_batch_dir / "archive_manifest.json",
        archive_root=archive_root,
        kept_run_ids=kept_run_ids,
        source_output_dir=output_dir,
        archived_run_entries=archived_run_entries,
        archived_loose_files=archived_loose_file_paths,
        deleted_transient_files=deleted_transient_file_paths,
    )
    _prune_empty_dirs(output_dir)

    print(f"\nArchived {len(selected_run_ids)} run(s) to {archive_batch_dir}")
    if loose_archive_files:
        print(f"Archived {len(loose_archive_files)} loose file(s)")
    if transient_delete_files:
        print(f"Deleted {len(transient_delete_files)} transient file(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Archive older SkillsExtraction run artifacts out of the output directory."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory containing run artifacts and checkpoints (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--archive-root",
        default=str(DEFAULT_ARCHIVE_ROOT),
        help=f"Archive root directory, ideally outside the repo (default: {DEFAULT_ARCHIVE_ROOT})",
    )
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=2,
        help="Keep this many most-recent run ids in place when --run-id is not specified (default: 2)",
    )
    parser.add_argument(
        "--keep-run-id",
        action="append",
        default=[],
        help="Always keep this run id in place; can be passed multiple times",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Archive only these specific run ids; can be passed multiple times",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move the files instead of printing a dry-run plan",
    )
    args = parser.parse_args()

    try:
        return archive_runs(
            output_dir=Path(args.output_dir).expanduser().resolve(),
            archive_root=Path(args.archive_root).expanduser().resolve(),
            keep_latest=max(0, args.keep_latest),
            keep_run_ids=args.keep_run_id,
            explicit_run_ids=args.run_id,
            apply=args.apply,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
