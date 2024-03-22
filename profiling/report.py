"""Analyze and visualize profiling results."""

import datetime
import os
import pstats
from argparse import ArgumentParser
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

parser = ArgumentParser()
parser.add_argument("--event", help="Github event name", default="none")
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
    "--max-benchmark", help="Max number of benchmarks to plot", type=int, default=5
)
args = parser.parse_args()

root_path = Path.cwd()
orig_prof_path = root_path / args.src_dir / "combined.prof"
orig_graph_path = root_path / args.src_dir / "combined.svg"
prof_path = root_path / args.dst_dir / "profiling.prof"
graph_path = root_path / args.dst_dir / "call-graph.svg"
report_path = root_path / args.dst_dir / "prof-report.csv"

# Copy the profiling results to the current folder
if orig_prof_path.exists():
    os.system(f"cp {orig_prof_path} {prof_path}")
else:
    raise FileNotFoundError(f"Cannot find the profiling file at {orig_prof_path}.")
if orig_graph_path.exists():
    os.system(f"cp {orig_graph_path} {graph_path}")
else:
    print(f"Cannot find the call graph at {orig_graph_path}.")


def read_prof_as_dataframe(prof_path: str | Path) -> pd.DataFrame:
    """Convert the ".prof" profiling result to a DataFrame."""

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


def process_df_columns(df: pd.DataFrame, path_filter: str = "") -> pd.DataFrame:
    """Process the DataFrame columns including file name, function name, etc."""

    # Split the filename, lineno, and function
    df[["filename", "lineno", "function"]] = df.pop(
        "filename:lineno(function)"
    ).str.extract(r"(.*):(.*?)\((.*)\)", expand=True)

    # Filter the paths by a regex pattern (usually the module name)
    if path_filter:
        df = df[df.filename.str.contains(path_filter)]

    # Remove the common prefix of the filenames
    i_diff = next(i for i, cs in enumerate(zip(*df.filename)) if len(set(cs)) > 1)
    df["filename"] = [s[i_diff:] for s in df.filename]

    # Add current time as timestamp
    dt = datetime.datetime.fromtimestamp(prof_path.stat().st_mtime)
    df["timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S")

    # Add a unique label for each function for plotting
    df["label"] = (
        df["filename"].str.extract(r"(\w+).py").squeeze() + "." + df["function"]
    )

    # Add GitHub event name for information
    df["event"] = args.event

    return df


def plot_profiling(df: pd.DataFrame, save_path: str | bool = False) -> None:
    """Plot the profiling results."""

    print(df)
    df.plot.barh(
        y=["tottime_percall", "cumtime_percall"], x="label", figsize=(20, 10)
    )  # type: ignore
    plt.ylabel("")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.legend(loc="lower right")

    if args.show:
        plt.show()

    if save_path:
        plt.savefig(save_path)


def benchmark(df: pd.DataFrame, n_runs: int) -> pd.DataFrame:
    """Benchmark the profiling results with `n_runs` previous runs."""

    # Key performance indicators of the functions
    kpis = ["tottime_percall", "cumtime_percall", "tottime", "cumtime"]
    bm = df.pivot_table(index="timestamp", columns="label", values=kpis)  # type: ignore

    # Find the functions with the highest time costs for each KPI
    labels = df[kpis].idxmax()
    bm = bm[list(map(tuple, labels.reset_index().values))]  # select the top functions
    bm.columns = bm.columns.map(lambda x: f"{x[1]}({x[0]})")  # label(KPI)

    return bm.iloc[-n_runs:]


def plot_and_check_benchmark(bm: pd.DataFrame, save_path: str | bool = False) -> None:
    """Plot the benchmark results and check the performance change."""

    # Plot benchmark results
    bm.T.plot.barh(figsize=(20, 10))
    plt.tight_layout()
    plt.legend(loc="lower right")

    if args.show:
        plt.show()

    if save_path:
        plt.savefig(save_path)

    # Check performance changes
    time_costs = bm.max(axis=1)  # "cumtime" for each run
    t_ratio = time_costs / time_costs.shift(1) - 1  # ratio of change in time cost
    latest_change = t_ratio.iloc[-1]
    assert latest_change < 1.05


df = read_prof_as_dataframe(prof_path)
df = process_df_columns(df, path_filter=args.path_filter)

if not args.no_save:
    df.to_csv(report_path, mode="a", header=not report_path.exists(), index=False)

plot_profiling(df, not args.no_save and root_path / args.dst_dir / "profiling.png")

# Read the whole report file for benchmarking
df = pd.read_csv(report_path, index_col="label", parse_dates=["timestamp"])
bm = benchmark(df, args.max_benchmark)
plot_and_check_benchmark(
    bm, not args.no_save and root_path / args.dst_dir / "benchmark.png"
)
