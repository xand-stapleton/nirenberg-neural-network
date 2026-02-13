import argparse
import shutil
from pathlib import Path

import wandb

ENTITY = "logml"
PROJECT = "Ainstein_kw"


def extract_hps_from_run(run):
    command = None

    # Try metadata args
    if hasattr(run, "metadata") and run.metadata:
        command = run.metadata.get("args", [])

    # If not in metadata, try config
    if not command:
        command = run.config.get("args", [])

    # Sometimes command is stored as a string
    if not command and run.config:
        command_str = run.config.get("command", "")
        if command_str:
            command = command_str.split()

    # Try program + summary args
    if not command:
        program = run.config.get("program", "")
        if program and run.summary:
            args_list = run.summary.get("_wandb", {}).get("args", [])
            if args_list:
                command = [program] + args_list

    # Parse --hps
    if command and isinstance(command, (list, tuple)):
        for i, arg in enumerate(command):
            if arg == "--hps" and i + 1 < len(command):
                hps = command[i + 1]
                hps = Path(hps).name
                if hps.endswith(".yaml"):
                    hps = hps[:-5]
                return hps

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Organize wandb run directories by --hps or by run.group"
    )
    parser.add_argument(
        "runs_dir",
        type=Path,
        help="Directory containing wandb run folders (e.g. foo-bar-123)",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--by-group",
        action="store_true",
        help="Organize runs by wandb run.group instead of --hps",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without modifying the filesystem",
    )

    args = parser.parse_args()
    runs_dir: Path = args.runs_dir.resolve()

    if not runs_dir.exists():
        raise FileNotFoundError(f"{runs_dir} does not exist")

    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    for run in runs:
        run_dir = runs_dir / run.name
        if not run_dir.exists():
            continue

        if args.by_group:
            key = run.group or "ungrouped"
        else:
            key = extract_hps_from_run(run)
            if not key:
                print(f"[SKIP] {run.name}: no --hps found")
                continue

        target_dir = runs_dir / key
        dest = target_dir / run.name

        action = "[DRY-RUN]" if args.dry_run else "[MOVE]"
        print(f"{action} {run.name} → {key}/")

        if args.dry_run:
            continue

        target_dir.mkdir(exist_ok=True)
        shutil.move(run_dir, dest)


if __name__ == "__main__":
    main()

