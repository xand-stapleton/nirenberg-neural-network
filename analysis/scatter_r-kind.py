import os
import argparse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import wandb
import re

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ENTITY = "logml"
PROJECT = "Ainstein_kw"
LOSS_KEY = "epoch/conformal_loss"
CACHE_PATH = "wandb_loss_cache.parquet"
CACHE_TTL_HOURS = 100  # Refresh cache after this many hours


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        nargs="*",
        default=[],
        help=(
            "W&B run groups to include (pooled). "
            "Provide one or more; if none, fetches all runs. "
            "Example: --groups Polchinski-2-1 Polchinski-3-1"
        ),
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Required W&B tags (all must be present)",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        help=(
            "Unix-style wildcard patterns to ignore when plotting. "
            "Matches against run_id, R_kind, and hyperparam_file. "
            "Does not modify the cache."
        ),
    )
    parser.add_argument(
        "--ignore-not",
        nargs="*",
        default=[],
        help=(
            "Unix-style wildcard patterns to include when plotting. "
            "Keeps rows matching run_id, R_kind, or hyperparam_file; "
            "drops all others. Does not modify the cache."
        ),
    )
    parser.add_argument(
        "--cache-all",
        action="store_true",
        help="If set, only cache runs matching group/tags. "
        "Default: cache matching runs and filter at runtime.",
    )

    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore TTL and refresh the cache from W&B now.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Concurrent threads for per-run fetching (default: up to 8).",
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on number of runs to process.",
    )

    parser.add_argument(
        "--update-cache-names",
        action="store_true",
        help=(
            "Persist normalized hyperparameter file names (basename without "
            "folders or .yaml) back to the cache."
        ),
    )

    parser.add_argument(
        "--order-by",
        choices=["r_kind", "loss"],
        default="loss",
        help=(
            "Ordering of R_kind–loss pairs in the plot: "
            "'r_kind' (default) or 'loss' (increasing min_loss)"
        ),
    )

    parser.add_argument(
        "--plot-stat",
        choices=["min", "mean", "mean_std", "all"],
        default="min",
        help=(
            "What to plot per label: "
            "min (min over runs), mean, mean_std (mean±std), or all"
        ),
    )

    parser.add_argument(
        "--unknown-right",
        action="store_false",
        help=(
            "Place entries with unknown expected existence (None) to the "
            "right and sort them by increasing loss for the chosen plot "
            "statistic."
        ),
    )

    parser.add_argument(
        "--unknown-band",
        action="store_false",
        help=(
            "Shade a vertical band behind entries with unknown expected "
            "existence (None) for clear indication in the plot."
        ),
    )

    parser.add_argument(
        "--unrealizable-band",
        action="store_false",
        help=(
            "Shade a faint red vertical band behind entries marked as "
            "unrealizable (expected_existence=False)."
        ),
    )

    parser.add_argument(
        "--realizable-band",
        action="store_false",
        help=(
            "Shade a faint green vertical band behind entries marked as "
            "realizable (expected_existence=True)."
        ),
    )

    parser.add_argument(
        "--transition-line",
        action="store_true",
        help=(
            "Draw a dashed vertical line at the boundary where known "
            "entries transition from realizable (True) to unrealizable "
            "(False) in the current ordering. Most meaningful when "
            "--order-by loss."
        ),
    )
    parser.add_argument(
        "--transition-line-pt",
        type=float,
        default=None,
        help=(
            "If provided, also draw a dashed horizontal line at this loss "
            "value (y-axis). Uses log scale; value must be > 0."
        ),
    )
    parser.add_argument(
        "--transition-band",
        nargs=2,
        type=float,
        default=None,
        help=(
            "If provided, draw a horizontal shaded band between two loss "
            "values (y-axis). Format: --transition-band LOW HIGH. "
            "Uses log scale; values must be > 0."
        ),
    )

    parser.add_argument(
        "--hline",
        action="store_true",
        help=(
            "Draw a horizontal line at the midpoint between the highest "
            "loss realizable case and the lowest loss unrealizable case."
        ),
    )
    parser.add_argument(
        "--hline-label",
        default="Threshold",
        help="Legend label for the horizontal line",
    )

    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Disable the plot title",
    )

    parser.add_argument(
        "--use-hyperparam-file",
        action="store_true",
        default=True,
        help=(
            "Plot hyperparameter file names instead of R_kind "
            "(default: True)"
        ),
    )

    parser.add_argument(
        "--use-r-kind",
        action="store_true",
        help="Plot R_kind instead of hyperparameter file names",
    )

    parser.add_argument(
        "--reject-round-r-kind",
        action="store_true",
        help="If set, exclude runs where R_kind == 'round' (or hyperparam_file == 'round' if R_kind is None)",
    )
    parser.add_argument(
        "--reject-zero-r-kind",
        action="store_true",
        help="If set, exclude runs where R_kind == 'zero' (or hyperparam_file == 'zero' if R_kind is None)",
    )
    parser.add_argument(
        "--realizable",
        nargs="*",
        default=[],
        help=(
            "Labels to mark as realizable (expected_existence=True). "
            "Example: --realizable sh_2_0 sh_3_0"
        ),
    )
    parser.add_argument(
        "--unrealizable",
        nargs="*",
        default=[],
        help=(
            "Labels to mark as unrealizable (expected_existence=False). "
            "Example: --unrealizable sh_1_0"
        ),
    )
    parser.add_argument(
        "--cache-files",
        nargs="+",
        default=None,
        help=(
            "Two or more cache files to pool together. "
            "When provided, these files are loaded and combined instead of "
            "fetching from W&B or using the default cache. "
            "Example: --cache-files cache1.parquet cache2.parquet"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------
def load_cached_df(ignore_ttl: bool = False):
    if not os.path.exists(CACHE_PATH):
        return None

    if not ignore_ttl:
        mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_PATH))
        if datetime.now() - mtime > timedelta(hours=CACHE_TTL_HOURS):
            print("Cache expired, refreshing from W&B API...")
            return None

    print(f"Loaded cached data from {CACHE_PATH}")
    return pl.read_parquet(CACHE_PATH)


def load_and_pool_cache_files(cache_files: list[str]):
    """Load multiple cache files and pool them into a single DataFrame."""
    if len(cache_files) < 2:
        raise ValueError("At least 2 cache files must be provided")
    
    dataframes = []
    for cache_file in cache_files:
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file does not exist: {cache_file}")
        
        print(f"Loading cache file: {cache_file}")
        df = pl.read_parquet(cache_file)
        print(f"  Loaded {len(df)} runs from {cache_file}")
        dataframes.append(df)
    
    # Pool all dataframes together
    pooled_df = pl.concat(dataframes)
    print(f"Pooled {len(pooled_df)} total runs from {len(cache_files)} cache files")
    
    # Remove duplicates based on run_id if any exist
    original_count = len(pooled_df)
    pooled_df = pooled_df.unique(subset=["run_id"])
    if len(pooled_df) < original_count:
        print(f"Removed {original_count - len(pooled_df)} duplicate runs")
    
    return pooled_df


# ---------------------------------------------------------------------
# W&B fetch
# ---------------------------------------------------------------------
def fetch_from_wandb(
    wandb_groups: list[str],
    required_tags: set[str],
    cache_only_matching: bool,
    threads: int,
    max_runs: int | None,
):
    print("Fetching runs from W&B API...")
    print(
        "Caching mode:",
        "ONLY matching runs" if cache_only_matching else "ALL runs",
    )
    api = wandb.Api()

    def runs_for_group(gr: str):
        # Prefer server-side filtering when caching only matching runs
        filters = {}
        if cache_only_matching:
            if gr:
                filters["group"] = gr
            if required_tags:
                # Require that all specified tags are present server-side
                filters["tags"] = {"$all": list(required_tags)}
        elif gr:
            # When caching broadly, still allow grouping, but avoid tag filter
            filters["group"] = gr

        return api.runs(
            f"{ENTITY}/{PROJECT}",
            filters=filters or None,
            per_page=200,
        )

    all_runs = []
    groups_list = list(wandb_groups or [])
    if groups_list:
        for gr in groups_list:
            all_runs.extend(list(runs_for_group(gr)))
    else:
        # Fallback: no specific groups provided; apply tag filter if strict
        base_filters = {}
        if cache_only_matching and required_tags:
            base_filters["tags"] = {"$all": list(required_tags)}
        all_runs.extend(
            list(
                api.runs(
                    f"{ENTITY}/{PROJECT}",
                    filters=base_filters or None,
                    per_page=200,
                )
            )
        )

    if max_runs is not None:
        all_runs = all_runs[:max_runs]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def extract_hps(run):
        hyperparam_file = "unknown"
        command = None
        if hasattr(run, 'metadata') and run.metadata:
            command = run.metadata.get('args', [])
        if not command:
            command = run.config.get('args', [])
        if not command and hasattr(run, 'config') and run.config:
            command_str = run.config.get('command', '')
            if command_str:
                command = command_str.split()
        if not command and hasattr(run, 'config'):
            program = run.config.get('program', '')
            if program and hasattr(run, 'summary') and run.summary:
                args_list = run.summary.get('_wandb', {}).get('args', [])
                if args_list:
                    command = [program] + args_list
        if command and isinstance(command, (list, tuple)):
            for i, arg in enumerate(command):
                if arg == '--hps' and i + 1 < len(command):
                    hyperparam_file = command[i + 1]
                    # Normalize: remove extension; take basename after slash
                    if hyperparam_file.endswith('.yaml'):
                        hyperparam_file = hyperparam_file[:-5]
                    # Strip leading known folder, then take last path segment
                    if hyperparam_file.startswith('configs/'):
                        hyperparam_file = hyperparam_file[8:]
                    if '/' in hyperparam_file:
                        hyperparam_file = hyperparam_file.split('/')[-1]
                    break
        return hyperparam_file

    def compute_min_loss(run):
        min_loss = None
        try:
            # Stream history to avoid large pandas DataFrame conversion
            for row in run.scan_history(keys=[LOSS_KEY], page_size=1000):
                v = row.get(LOSS_KEY)
                if v is None:
                    continue
                try:
                    if min_loss is None or float(v) < float(min_loss):
                        min_loss = float(v)
                except Exception:
                    continue
        except Exception as e:
            rid = getattr(run, 'id', '?')
            print(f"History scan failed for run {rid}: {e}")
        return min_loss

    def process_run(run):
        # Client-side filtering if requested (still avoids artifacts)
        if cache_only_matching:
            if wandb_groups:
                if run.group not in wandb_groups:
                    return None
            if required_tags:
                run_tags = set(run.tags or [])
                if not required_tags.issubset(run_tags):
                    return None

        ml = compute_min_loss(run)
        if ml is None:
            return None

        expected_existence = run.config.get("data", {}).get(
            "expected_existence", None
        )
        prescribed_R = run.config.get("data", {}).get("prescribed_R")
        if isinstance(prescribed_R, dict):
            r_kind = prescribed_R.get("kind")
        else:
            r_kind = prescribed_R

        hyperparam_file = extract_hps(run)

        return {
            "run_id": run.id,
            "group": run.group,
            "tags": list(run.tags or []),
            "R_kind": r_kind,
            "hyperparam_file": hyperparam_file,
            "expected_existence": expected_existence,
            "min_loss": ml,
        }

    records = []
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        futures = {ex.submit(process_run, run): run for run in all_runs}
        for fut in as_completed(futures):
            rec = fut.result()
            if rec is not None:
                records.append(rec)

    df = pl.DataFrame(records)
    df.write_parquet(CACHE_PATH)
    print(f"Cached {len(df)} runs to {CACHE_PATH}")
    return df


# ---------------------------------------------------------------------
# Runtime filtering
# ---------------------------------------------------------------------
def filter_df(
    df: pl.DataFrame,
    wandb_groups: list[str],
    required_tags: set[str],
):
    if wandb_groups:
        df = df.filter(pl.col("group").is_in(wandb_groups))

    if required_tags:
        for tag in required_tags:
            df = df.filter(pl.col("tags").list.contains(tag))

    print(df)
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    wandb_groups = list(args.groups or [])
    required_tags = set(args.tags)

    # Handle multiple cache files if provided
    if args.cache_files:
        if len(args.cache_files) < 2:
            print("Error: At least 2 cache files must be provided with --cache-files")
            exit(1)
        
        print(f"Loading and pooling {len(args.cache_files)} cache files...")
        try:
            df = load_and_pool_cache_files(args.cache_files)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading cache files: {e}")
            exit(1)
    else:
        # Original single cache file logic
        # If overrides are provided, prefer loading cache even if TTL expired,
        # so we can update and persist without contacting W&B.
        ignore_ttl = bool(args.realizable or args.unrealizable)
        df = None if args.refresh_cache else load_cached_df(ignore_ttl=ignore_ttl)
        if df is None:
            df = fetch_from_wandb(
                wandb_groups,
                required_tags,
                cache_only_matching=not (args.cache_all),
                threads=args.threads,
                max_runs=args.max_runs,
            )

    # -----------------------------
    # Apply runtime filters
    # -----------------------------
    df = filter_df(df, wandb_groups, required_tags)
    # -----------------------------
    # Normalize kind column: prefer R_kind, fallback to K_kind
    # -----------------------------
    try:
        cols = set(df.columns)
        if "R_kind" not in cols and "K_kind" in cols:
            df = df.rename({"K_kind": "R_kind"})
        elif "R_kind" in cols and "K_kind" in cols:
            df = df.with_columns(
                pl.coalesce([
                    pl.col("R_kind"),
                    pl.col("K_kind"),
                ]).alias("R_kind")
            )
        if "R_kind" in df.columns:
            # Standardize to lowercase trimmed strings for robust matching
            df = df.with_columns(
                pl.when(pl.col("R_kind").is_null())
                .then(pl.lit(None))
                .otherwise(
                    pl.col("R_kind").cast(
                        pl.Utf8).str.strip().str.to_lowercase()
                )
                .alias("R_kind")
            )
    except Exception:
        pass
    # -----------------------------
    # Normalize hyperparam_file values (basename, no .yaml)
    # -----------------------------
    def map_thm_to_prop(name: str) -> str:
        """Map thm_* names to prop_* for display while keeping thm_* on wandb."""
        if name is None:
            return name
        mapping = {
            "thm_a": "prop_a",
            "thm_b": "prop_b", 
            "thm_c": "prop_c"
        }
        return mapping.get(name, name)
    
    df = df.with_columns(
        [
            pl.when(pl.col("hyperparam_file").is_not_null())
            .then(
                pl.col("hyperparam_file")
                .cast(pl.Utf8)
                .str.split("/")
                .list.last()
                .str.replace(r"\\.yaml$", "")
            )
            .otherwise(pl.col("hyperparam_file"))
            .alias("hyperparam_file")
        ]
    )
    
    # Map thm_* to prop_* for display purposes
    df = df.with_columns(
        pl.col("hyperparam_file").map_elements(
            lambda x: map_thm_to_prop(x), 
            return_dtype=pl.Utf8
        ).alias("hyperparam_file")
    )

    if args.update_cache_names:
        if args.cache_files:
            print("Warning: --update-cache-names ignored when using --cache-files (multiple cache files)")
        else:
            df.write_parquet(CACHE_PATH)
            print("Updated cache with normalized hyperparameter file names.")

    if df.is_empty():
        print("No runs found matching the filters.")
        exit()

    # -----------------------------
    # Determine which field to use
    # -----------------------------
    use_hyperparam_file = args.use_hyperparam_file and not args.use_r_kind
    plot_field = "hyperparam_file" if use_hyperparam_file else "R_kind"
    plot_label = "Prescriber" if use_hyperparam_file else "R_kind"

    print(f"Plotting by: {plot_label}")

    # -----------------------------
    # Apply existence overrides and persist to cache
    # -----------------------------
    realizable = set(args.realizable or [])
    unrealizable = set(args.unrealizable or [])
    if realizable or unrealizable:
        if realizable:
            df = df.with_columns(
                pl.when(pl.col(plot_field).is_in(list(realizable)))
                .then(pl.lit(True))
                .otherwise(pl.col("expected_existence"))
                .alias("expected_existence")
            )
        if unrealizable:
            df = df.with_columns(
                pl.when(pl.col(plot_field).is_in(list(unrealizable)))
                .then(pl.lit(False))
                .otherwise(pl.col("expected_existence"))
                .alias("expected_existence")
            )
        # Persist updated existence flags back to cache without new API calls.
        if args.cache_files:
            print("Warning: Cannot update cache when using --cache-files (multiple cache files)")
            print("Applied existence overrides to current session only.")
        else:
            df.write_parquet(CACHE_PATH)
            print(
                "Updated cache with existence overrides (no API requests)."
            )

    # -----------------------------
    # Optional R_kind rejection
    # -----------------------------
    if "R_kind" in df.columns:
        # Debug: show unique R_kind values before filtering
        unique_r_kinds = df.select(
            pl.col("R_kind")).unique().to_series().to_list()
        print(f"Unique R_kind values before reject filters: {unique_r_kinds}")

        # Check if R_kind has meaningful values or if we should fall back to hyperparam_file
        has_meaningful_r_kind = any(x is not None for x in unique_r_kinds)

        original_count = len(df)

        if args.reject_round_r_kind:
            if has_meaningful_r_kind:
                # Filter on R_kind
                df = df.filter(
                    ~pl.col("R_kind").cast(pl.Utf8).fill_null(
                        "").str.to_lowercase().is_in(["round"])
                )
                print(
                    f"After reject-round-r-kind (R_kind): {len(df)} rows (removed {original_count - len(df)})")
            else:
                # Fallback to hyperparam_file
                df = df.filter(
                    ~pl.col("hyperparam_file").cast(pl.Utf8).fill_null(
                        "").str.to_lowercase().is_in(["round"])
                )
                print(
                    f"After reject-round-r-kind (hyperparam_file): {len(df)} rows (removed {original_count - len(df)})")
            original_count = len(df)

        if args.reject_zero_r_kind:
            if has_meaningful_r_kind:
                # Filter on R_kind
                df = df.filter(
                    ~pl.col("R_kind").cast(pl.Utf8).fill_null(
                        "").str.to_lowercase().is_in(["zero"])
                )
                print(
                    f"After reject-zero-r-kind (R_kind): {len(df)} rows (removed {original_count - len(df)})")
            else:
                # Fallback to hyperparam_file
                df = df.filter(
                    ~pl.col("hyperparam_file").cast(pl.Utf8).fill_null(
                        "").str.to_lowercase().is_in(["zero"])
                )
                print(
                    f"After reject-zero-r-kind (hyperparam_file): {len(df)} rows (removed {original_count - len(df)})")

    # -----------------------------
    # Optional ignore patterns (wildcards) - runtime only
    # -----------------------------
    ignore_patterns = list(args.ignore or [])
    if ignore_patterns:
        def _wild_to_regex(p: str) -> str:
            esc = re.escape(p)
            return "^" + esc.replace("\\*", ".*").replace("\\?", ".") + "$"

        regexes = [_wild_to_regex(p) for p in ignore_patterns]
        # Build mask: rows to drop if any field matches any pattern
        masks = []
        for rx in regexes:
            masks.append(
                pl.col("run_id").fill_null("").str.contains(rx)
                |
                pl.col("hyperparam_file").fill_null("").str.contains(rx)
                |
                pl.col("R_kind").cast(pl.Utf8).fill_null("").str.contains(rx)
            )
        if masks:
            ignore_mask = masks[0]
            for m in masks[1:]:
                ignore_mask = ignore_mask | m
            df = df.filter(~ignore_mask)

    # -----------------------------
    # Optional inverse ignore (wildcards) - include-only runtime filter
    # -----------------------------
    ignore_not_patterns = list(args.ignore_not or [])
    if ignore_not_patterns:
        def _wild_to_regex_not(p: str) -> str:
            esc = re.escape(p)
            return "^" + esc.replace("\\*", ".*").replace("\\?", ".") + "$"

        regexes_not = [_wild_to_regex_not(p) for p in ignore_not_patterns]
        include_masks = []
        for rx in regexes_not:
            include_masks.append(
                pl.col("run_id").fill_null("").str.contains(rx)
                |
                pl.col("hyperparam_file").fill_null("").str.contains(rx)
                |
                pl.col("R_kind").cast(pl.Utf8).fill_null("").str.contains(rx)
            )
        if include_masks:
            include_mask = include_masks[0]
            for m in include_masks[1:]:
                include_mask = include_mask | m
            df = df.filter(include_mask)

    # -----------------------------
    # Aggregate stats per chosen field
    # -----------------------------
    df_stats = df.group_by(plot_field).agg(
        [
            pl.col("min_loss").min().alias("min_loss"),
            pl.col("min_loss").mean().alias("mean_loss"),
            pl.col("min_loss").std().alias("std_loss"),
            pl.col("min_loss").count().alias("count"),
            pl.col("expected_existence").first().alias("expected_existence"),
        ]
    )

    # std can be null when there is only 1 run for a label
    df_stats = df_stats.with_columns(pl.col("std_loss").fill_null(0.0))

    # Apply requested ordering
    if args.order_by == "loss":
        if args.plot_stat in {"mean", "mean_std"}:
            df_plot = df_stats.sort("mean_loss")
        else:
            df_plot = df_stats.sort("min_loss")
    else:  # args.order_by == "r_kind"
        df_plot = df_stats.sort(plot_field)

    # Optionally move unknown expected_existence to the right, sorted by loss
    if args.unknown_right:
        stat_key = (
            "mean_loss" if args.plot_stat in {"mean", "mean_std", "all"}
            else "min_loss"
        )
        known = df_plot.filter(
            pl.col("expected_existence").is_in([True, False])
        )
        unknown = df_plot.filter(
            pl.col("expected_existence").is_null()
        )
        unknown_sorted = unknown.sort(stat_key)
        df_plot = pl.concat([known, unknown_sorted])

    print(
        f"Found {len(df_plot)} unique {plot_label} values "
        f"from {len(df)} total runs"
    )

    # Overrides already applied to df and persisted; df_stats reflects them.

    # -----------------------------
    # Report lowest-loss runs per hyperparam file
    # -----------------------------
    try:
        df_hp = df.select(
            ["hyperparam_file", "run_id", "min_loss"]
        ).drop_nulls("hyperparam_file")
        if not df_hp.is_empty():
            hp_mins = df_hp.group_by("hyperparam_file").agg(
                pl.col("min_loss").min().alias("min_loss")
            )
            best_runs = df_hp.join(
                hp_mins,
                on=["hyperparam_file", "min_loss"],
                how="inner",
            )
            summary = best_runs.group_by("hyperparam_file").agg([
                pl.col("min_loss").first().alias("min_loss"),
                pl.col("run_id").unique().alias("run_ids"),
            ]).sort("min_loss")

            print("\nLowest-loss runs per hyperparam file:")
            for rec in summary.iter_rows(named=True):
                hp = rec["hyperparam_file"]
                loss = rec["min_loss"]
                runs = rec["run_ids"]
                loss_str = f"{float(loss):.6g}" if loss is not None else "NA"
                if isinstance(runs, list):
                    runs_str = ", ".join(runs)
                else:
                    runs_str = str(runs)
                print(f"  {hp}: loss={loss_str}, runs={runs_str}")
        else:
            print(
                "No hyperparam_file information available to "
                "report best runs."
            )
    except Exception as e:
        print(f"Failed to summarize lowest-loss runs per hyperparam file: {e}")

    # -----------------------------
    # Plot
    # -----------------------------
    plot_labels = df_plot[plot_field].to_list()
    x_positions = np.arange(len(plot_labels))

    plt.figure(figsize=(9, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(df_plot)))

    mins = np.array([float(v) for v in df_plot["min_loss"]], dtype=float)
    means = np.array([float(v) for v in df_plot["mean_loss"]], dtype=float)
    stds = np.array([float(v) for v in df_plot["std_loss"]], dtype=float)

    # Track a reference y per-point for placing symbols later.
    y_for_symbols = mins.copy()

    if args.plot_stat == "min":
        plt.scatter(
            x_positions,
            mins,
            c=colors,
            s=90,
            alpha=0.85,
            edgecolor="black",
            label="Min. prescriber loss",
        )
        y_for_symbols = mins
        title_prefix = "Min. prescriber loss"
    elif args.plot_stat == "mean":
        plt.scatter(
            x_positions,
            means,
            c=colors,
            s=90,
            alpha=0.85,
            edgecolor="black",
            marker="D",
            label="Mean loss",
        )
        y_for_symbols = means
        title_prefix = "Mean Loss"
    elif args.plot_stat == "mean_std":
        # On log scale, keep lower error bars above 0.
        eps = 1e-12
        lower_err = np.minimum(stds, np.maximum(means - eps, 0.0))
        upper_err = stds
        yerr = np.vstack([lower_err, upper_err])
        plt.errorbar(
            x_positions,
            means,
            yerr=yerr,
            fmt="o",
            ecolor="black",
            elinewidth=1.0,
            capsize=4,
            capthick=1.0,
            markersize=7,
            markeredgecolor="black",
            markerfacecolor="white",
            alpha=0.9,
            label="Mean ± Std loss",
        )
        y_for_symbols = means + stds
        title_prefix = "Mean ± Std Loss"
    else:  # args.plot_stat == "all"
        eps = 1e-12
        lower_err = np.minimum(stds, np.maximum(means - eps, 0.0))
        upper_err = stds
        yerr = np.vstack([lower_err, upper_err])
        plt.errorbar(
            x_positions,
            means,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=4,
            capthick=1.0,
            alpha=0.7,
            label="Std loss",
        )
        plt.scatter(
            x_positions,
            mins,
            c=colors,
            s=90,
            alpha=0.85,
            edgecolor="black",
            label="Min. prescriber loss",
        )
        plt.scatter(
            x_positions,
            means,
            c=colors,
            s=70,
            alpha=0.9,
            edgecolor="black",
            marker="D",
            label="Mean loss",
        )
        y_for_symbols = np.maximum(mins, means + stds)
        title_prefix = "Min / Mean / Std Loss"

    plt.yscale("log")
    plt.xticks(x_positions, plot_labels, rotation=90)
    plt.xlabel(plot_label)
    plt.ylabel("Loss (log scale)")
    if not args.no_title:
        groups_str = ", ".join(wandb_groups)
        tags_str = ", ".join(sorted(required_tags)) or "none"
        plt.title(
            f"{title_prefix} per {plot_label}\n"
            f"Groups: {groups_str}, Tags: {tags_str}"
        )
    plt.grid(True, alpha=0.3)

    # Key for what is being plotted (controlled by --plot-stat)
    if args.plot_stat == "all":
        plt.legend(title="Legend", loc="upper left", frameon=True)
    else:
        # For single-stat plots, still show a small legend so the PDF
        # is self-describing.
        plt.legend(loc="upper left", frameon=True)

    # Give a bit more headroom above the top point(s)
    ax = plt.gca()
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 1.8)

    # -----------------------------
    # Unknown band shading (optional)
    # -----------------------------
    def _unknown_spans(indices: list[int]):
        spans = []
        if not indices:
            return spans
        start = indices[0]
        prev = start
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                spans.append((start, prev))
                start = idx
                prev = idx
        spans.append((start, prev))
        return spans

    if args.unknown_band or args.unknown_right:
        unknown_indices = [
            i for i, v in enumerate(df_plot["expected_existence"]) if v is None
        ]
        spans = _unknown_spans(unknown_indices)
        for j, (a, b) in enumerate(spans):
            left = float(a) - 0.5
            right = float(b) + 0.5
            ax.axvspan(
                left,
                right,
                color="lightgray",
                alpha=0.4,
                zorder=0.5,
            )

    # Realizable / Unrealizable band shading (optional)
    if args.realizable_band:
        realizable_indices = [
            i for i, v in enumerate(df_plot["expected_existence"]) if v is True
        ]
        spans = _unknown_spans(realizable_indices)
        for j, (a, b) in enumerate(spans):
            left = float(a) - 0.5
            right = float(b) + 0.5
            ax.axvspan(
                left,
                right,
                color="green",
                alpha=0.12,
                zorder=0.3,
            )

    if args.unrealizable_band:
        unrealizable_indices = [
            i for i, v in enumerate(
                df_plot["expected_existence"]
            ) if v is False
        ]
        spans = _unknown_spans(unrealizable_indices)
        for j, (a, b) in enumerate(spans):
            left = float(a) - 0.5
            right = float(b) + 0.5
            ax.axvspan(
                left,
                right,
                color="red",
                alpha=0.12,
                zorder=0.3,
            )

    # -----------------------------
    # Transition line (optional)
    # -----------------------------
    if args.transition_line:
        # Find first boundary from realizable (True) to unrealizable (False),
        # ignoring unknowns (None), in the current df_plot order.
        states = [
            None if v is None else (True if v else False)
            for v in df_plot["expected_existence"]
        ]
        last = None
        boundary_x = None
        for i, st in enumerate(states):
            if st is None:
                continue
            if last is None:
                last = st
                continue
            if last is True and st is False:
                boundary_x = float(i) - 0.5
                break
            last = st

        # if boundary_x is not None:
        #     ax.axvline(
        #         boundary_x,
        #         color="red",
        #         linestyle="--",
        #         linewidth=1.2,
        #         alpha=0.7,
        #         label="Realizable → Unrealizable",
        #     )

    # Optional horizontal transition line at a specific loss value
    if args.transition_line_pt is not None:
        try:
            y_val = float(args.transition_line_pt)
            if y_val <= 0 and plt.gca().get_yscale() == "log":
                print(
                    "Warning: transition-line-pt <= 0 on log scale; skipping."
                )
            else:
                ax.axhline(
                    y_val,
                    color="red",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.7,
                    label="Transition midpoint",
                )
        except Exception as e:
            print(f"Failed to draw horizontal transition line: {e}")

    # Optional horizontal transition band between two loss values
    if args.transition_band is not None:
        try:
            low_val, high_val = args.transition_band
            low_val = float(low_val)
            high_val = float(high_val)
            
            if low_val <= 0 or high_val <= 0:
                if plt.gca().get_yscale() == "log":
                    print(
                        "Warning: transition-band values <= 0 on log scale; skipping."
                    )
                else:
                    ax.axhspan(
                        min(low_val, high_val),
                        max(low_val, high_val),
                        color="orange",
                        alpha=0.2,
                        zorder=0.5,
                        label="Transition band",
                    )
            else:
                ax.axhspan(
                    min(low_val, high_val),
                    max(low_val, high_val),
                    color="orange",
                    alpha=0.2,
                    zorder=0.5,
                    label="Transition band",
                )
        except Exception as e:
            print(f"Failed to draw horizontal transition band: {e}")

    # -----------------------------
    # Horizontal line (optional)
    # -----------------------------
    if args.hline:
        try:
            stat_key = (
                "mean_loss" if args.plot_stat in {"mean", "mean_std", "all"}
                else "min_loss"
            )
            known_true = df_plot.filter(
                pl.col("expected_existence").eq(True)
            )
            known_false = df_plot.filter(
                pl.col("expected_existence").eq(False)
            )

            true_vals = [
                float(v)
                for v in known_true[stat_key].to_list()
                if v is not None
            ]
            false_vals = [
                float(v)
                for v in known_false[stat_key].to_list()
                if v is not None
            ]

            if not true_vals or not false_vals:
                print(
                    "Warning: cannot draw hline; need both realizable "
                    "and unrealizable entries."
                )
            else:
                max_real = max(true_vals)
                min_unreal = min(false_vals)
                y_val = 0.5 * (max_real + min_unreal)

                if y_val <= 0 and plt.gca().get_yscale() == "log":
                    print(
                        "Warning: computed hline <= 0 on log scale; "
                        "skipping."
                    )
                else:
                    ax.axhline(
                        y_val,
                        color="gray",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.7,
                        label=args.hline_label,
                    )
        except Exception as e:
            print(f"Failed to draw horizontal line: {e}")

    # -----------------------------
    # Expected existence symbols
    # -----------------------------
    for x, y, value in zip(
        x_positions,
        y_for_symbols,
        df_plot["expected_existence"],
    ):
        if value is True:
            symbol, color = "✓", "green"
        elif value is False:
            symbol, color = "✗", "red"
        else:
            symbol, color = "?", "gray"

        plt.text(
            float(x),
            1.3 * float(y),
            symbol,
            ha="center",
            va="bottom",
            fontsize=12,
            color=color,
        )

    # Refresh legend to include bands/lines added after the first legend
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    base_name = "hyperparam_file" if use_hyperparam_file else "R_kind"
    filename = f"{base_name}_{args.plot_stat}.pdf"
    plt.savefig(filename)
    plt.show()
