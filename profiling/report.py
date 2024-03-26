"""Analyze and visualize profiling results."""

import datetime
import os
import pstats
from argparse import ArgumentParser, Namespace
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_prof_as_dataframe(prof_path: str | Path) -> pd.DataFrame:
    """Convert the ".prof" profiling result to a DataFrame.

    Args:
        prof_path: Path to the profiling result.
    """

    # Read the profiling results
    sio = StringIO()
    p = pstats.Stats(str(prof_path), stream=sio)
    p.print_stats("pyrealm")
    sio = StringIO(sio.getvalue())

    # Convert to a DataFrame
    df = (
        pd.read_csv(sio, sep=" +", engine="python", skiprows=7)
        .rename(columns={"percall": "tottime_percall", "percall.1": "cumtime_percall"})
        .sort_values(by="cumtime_percall", ascending=False)
        .query("cumtime > 0.01")  # drop insignificant functions
        .dropna()
    )

    return df


def process_report_columns(df: pd.DataFrame, cfg: Namespace) -> pd.DataFrame:
    """Process the columns of the DataFrame report.

    Args:
        df: Profiling report DataFrame.
        cfg: Configuration of the profiling report.
    """

    # Split the filename, lineno, and function
    df[["filename", "lineno", "function"]] = df.pop(
        "filename:lineno(function)"
    ).str.extract(r"(.*):(.*?)\((.*)\)", expand=True)

    # Filter the paths by a regex pattern (usually the module name)
    if cfg.path_filter:
        df = df[df.filename.str.contains(cfg.path_filter)]

    # Remove the common prefix of the filenames
    i_diff = next(i for i, cs in enumerate(zip(*df.filename)) if len(set(cs)) > 1)
    df["filename"] = [s[i_diff:] for s in df.filename]

    # Add timestamp of profiling
    dt = datetime.datetime.fromtimestamp(cfg.profile_time)
    df = df.assign(timestamp=dt.strftime("%Y-%m-%d %H:%M:%S")).set_index("timestamp")

    # Add a unique label for each function for plotting
    df["label"] = (
        df["filename"].str.extract(r"(\w+).py").squeeze() + "." + df["function"]
    )

    # Add GitHub event name for information
    df["event"] = cfg.event

    if not cfg.no_save:
        df.to_csv(cfg.report_path, mode="a", header=not cfg.report_path.exists())

    return df


def plot_profiling(df: pd.DataFrame, cfg: Namespace) -> None:
    """Plot the profiling results.

    Args:
        df: Profiling report DataFrame.
        cfg: Configuration of the profiling report.
    """

    df.plot.barh(
        y=["tottime_percall", "cumtime_percall"], x="label", figsize=(20, 10)
    )  # type: ignore
    plt.ylabel("")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.legend(loc="lower right")

    if cfg.show:
        plt.show()

    if cfg.prof_plot_path:
        plt.savefig(cfg.prof_plot_path)


def run_benchmark(df: pd.DataFrame, cfg: Namespace) -> pd.DataFrame:
    """Benchmark the profiling results with `n_runs` previous runs.

    Args:
        df: Profiling report DataFrame.
        cfg: Configuration of the profiling report.
    """

    # Key performance indicators of the functions
    kpis = ["tottime_percall", "cumtime_percall", "tottime", "cumtime"]
    bm = df.pivot_table(index="timestamp", columns="label", values=kpis)  # type: ignore

    # Find the functions with the highest time costs for each KPI
    labels = df[kpis].idxmax()
    bm = bm[list(map(tuple, labels.reset_index().values))]  # select the top functions
    bm.columns = bm.columns.map(lambda x: f"{x[1]}({x[0]})")  # label(KPI)

    # Check performance changes
    time_costs = bm.max(axis=1)  # "cumtime" for each run
    t_ratio = time_costs / time_costs.shift(1) - 1  # ratio of change in time cost
    latest_change = t_ratio.iloc[-1]
    assert latest_change < 1 + cfg.bm_tolerance, "Performance is getting worse!"

    return bm.iloc[-cfg.bm_runs :]


def plot_benchmark(bm: pd.DataFrame, cfg: Namespace) -> None:
    """Plot the benchmark results and check the performance change.

    Args:
        bm: Benchmark DataFrame.
        cfg: Configuration of the profiling report.
    """

    # Plot benchmark results
    bm.T.plot.barh(figsize=(20, 10))
    plt.tight_layout()
    plt.legend(loc="lower right")

    if cfg.show:
        plt.show()

    if cfg.bm_plot_path:
        plt.savefig(cfg.bm_plot_path)


def profile_report_cli() -> None:
    """Generate the profiling report."""

    parser = ArgumentParser()
    parser.add_argument(
        "--src-dir", help="Directory of the generated profiling results", default="prof"
    )
    parser.add_argument(
        "--dst-dir", help="Directory to save the profiling report", default="profiling"
    )
    parser.add_argument(
        "--no-save", help="Do not save the profiling results", action="store_true"
    )
    parser.add_argument(
        "--show", help="Show the plots (which blocks the program)", action="store_true"
    )
    parser.add_argument(
        "--path-filter", help="Filter the paths by regex, e.g. 'splash', 'pmodel'"
    )
    parser.add_argument(
        "--bm-runs", help="Max number of latest runs to benchmark", type=int, default=5
    )
    parser.add_argument(
        "--bm-tolerance",
        help="Tolerance of time cost increase in benchmarking",
        type=float,
        default=0.05,
    )
    parser.add_argument("--event", help="Github event name", default="none")
    cfg = parser.parse_args()

    root_path = Path.cwd()
    orig_prof_path = root_path / cfg.src_dir / "combined.prof"
    orig_graph_path = root_path / cfg.src_dir / "combined.svg"
    prof_path = root_path / cfg.dst_dir / "profiling.prof"
    graph_path = root_path / cfg.dst_dir / "call-graph.svg"
    report_path = root_path / cfg.dst_dir / "prof-report.csv"

    # Add saving paths into the config
    cfg.report_path = report_path
    cfg.prof_plot_path = not cfg.no_save and root_path / cfg.dst_dir / "profiling.png"
    cfg.bm_plot_path = not cfg.no_save and root_path / cfg.dst_dir / "benchmark.png"
    cfg.profile_time = orig_prof_path.stat().st_mtime

    # Copy the profiling results to the current folder
    if orig_prof_path.exists():
        os.system(f"cp {orig_prof_path} {prof_path}")
    else:
        raise FileNotFoundError(f"Cannot find the profiling file at {orig_prof_path}.")
    if orig_graph_path.exists():
        os.system(f"cp {orig_graph_path} {graph_path}")
    else:
        print(f"Cannot find the call graph at {orig_graph_path}.")

    df = read_prof_as_dataframe(prof_path)
    df = process_report_columns(df, cfg)

    plot_profiling(df, cfg)

    # Read the whole report file for benchmarking
    df = pd.read_csv(report_path, index_col="label", parse_dates=["timestamp"])
    bm = run_benchmark(df, cfg)
    plot_benchmark(bm, cfg)


if __name__ == "__main__":
    profile_report_cli()
    print("Profiling report generated successfully.")
