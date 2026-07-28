#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a symlinked UniWM dataset subset from eval_dataset_manifest.json."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cfg/eval_dataset_manifest.json"),
        help="Path to manifest JSON.",
    )
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing the full UniWM datasets.",
    )
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=Path("eval_data"),
        help="Output root for the symlinked subset dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory first if it already exists.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=False,
        help="Copy directories instead of symlinking. Uses much more disk space.",
    )

    return parser.parse_args()

SPLITS = ("train", "validation", "test")


def load_manifest(path: Path) -> dict[str, dict[str, list[str]]]:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    return {
        name: entry
        for name, entry in manifest.items()
        if name != "action_token_vocabulary"
    }


def build_episode_index(dataset_root: Path, target_names: set[str]) -> dict[str, Path]:
    """
    Index only directories that:
      1. match one of the manifest episode names, and
      2. contain traj_data.pkl.

    This avoids assuming that episode directories are named traj_*.
    """
    index: dict[str, Path] = {}

    for candidate in dataset_root.rglob("*"):
        if not candidate.is_dir():
            continue

        if candidate.name not in target_names:
            continue

        if not (candidate / "traj_data.pkl").exists():
            continue

        if candidate.name in index:
            raise RuntimeError(
                f"Duplicate episode directory named {candidate.name!r} found:\n"
                f"  existing: {index[candidate.name]}\n"
                f"  duplicate: {candidate}"
            )

        index[candidate.name] = candidate.resolve()

    return index


def link_or_copy_episode(src: Path, dst: Path, use_copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Output path already exists: {dst}")

    if use_copy:
        shutil.copytree(src, dst)
    else:
        # Relative symlinks keep the output folder more portable if the repo moves as a unit.
        rel_src = Path("../" * len(dst.parent.relative_to(dst.anchor).parts)) / src
        try:
            dst.symlink_to(src, target_is_directory=True)
        except OSError:
            # Fall back to absolute symlink. This is usually fine on Linux workstations.
            dst.symlink_to(src, target_is_directory=True)


def main() -> None:
    args = parse_args()

    manifest_path = args.manifest.resolve()
    source_data_dir = args.source_data_dir.resolve()
    output_data_dir = args.output_data_dir.resolve()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not source_data_dir.exists():
        raise FileNotFoundError(f"Source data dir not found: {source_data_dir}")

    if output_data_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_data_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_data_dir}\n"
                f"Rerun with --overwrite if you want to recreate it."
            )

    manifest = load_manifest(manifest_path)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    missing: dict[str, list[str]] = {}
    manifest = load_manifest(manifest_path)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    references_by_dataset = {
        dataset_name: [
            reference
            for split in SPLITS
            for reference in entry[split]
        ]
        for dataset_name, entry in manifest.items()
    }

    needed_by_source: dict[str, set[str]] = {}

    for dataset_name, references in references_by_dataset.items():
        for reference in references:
            if "/" in reference:
                source, trajectory_id = reference.split("/", 1)
            else:
                source, trajectory_id = dataset_name, reference

            needed_by_source.setdefault(source, set()).add(trajectory_id)

    episode_indices: dict[str, dict[str, Path]] = {}

    for source, requested in needed_by_source.items():
        dataset_root = source_data_dir / source

        if not dataset_root.exists():
            print(f"[MISSING DATASET] {source}: {dataset_root}")
            episode_indices[source] = {}
            continue

        print(f"[INDEX] {source}: scanning for {len(requested)} trajectories")
        episode_indices[source] = build_episode_index(
            dataset_root,
            requested,
        )

    grand_total = 0
    missing: dict[str, list[str]] = {
        dataset_name: []
        for dataset_name in manifest
    }

    for dataset_name, references in references_by_dataset.items():
        out_dataset_root = output_data_dir / dataset_name
        out_dataset_root.mkdir(parents=True, exist_ok=True)

        linked_count = 0

        for reference in references:
            if "/" in reference:
                source, trajectory_id = reference.split("/", 1)
            else:
                source, trajectory_id = dataset_name, reference

            src = episode_indices.get(source, {}).get(trajectory_id)

            if src is None:
                missing[dataset_name].append(reference)
                continue

            # Qualified bootstrap references retain their source directory:
            # eval_data/bootstrap/go_stanford/<trajectory>
            dst = out_dataset_root / reference
            dst.parent.mkdir(parents=True, exist_ok=True)

            link_or_copy_episode(src, dst, use_copy=args.copy)
            linked_count += 1

        grand_total += linked_count
        print(
            f"[DONE] {dataset_name}: "
            f"linked/copied {linked_count}/{len(references)} trajectories"
        )
        
    total_missing = sum(len(v) for v in missing.values())
    if total_missing:
        print()
        print(f"[WARNING] Missing {total_missing} requested episodes:")
        for dataset_name, names in missing.items():
            if not names:
                continue
            print(f"  {dataset_name}: {len(names)} missing")
            for name in names[:20]:
                print(f"    - {name}")
            if len(names) > 20:
                print(f"    ... {len(names) - 20} more")
        raise SystemExit(1)

    print("[OK] All requested episodes were found.")


if __name__ == "__main__":
    main()
