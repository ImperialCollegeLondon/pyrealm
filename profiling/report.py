"""Analyze and visualize profiling results."""

import datetime
import os
import pstats
import re
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

root = Path.cwd()
orig_prof_path = root / "prof/combined.prof"
prof_path = root / "profiling/profiling.prof"
orig_graph_path = root / "prof/combined.svg"
graph_path = root / "profiling/call-graph.svg"

# Copy the profiling results to the current folder
if orig_prof_path.exists():
    os.system(f"cp {orig_prof_path} {prof_path}")
else:
    print(f"Cannot find the profiling file at {orig_prof_path}.")
    exit()
if orig_graph_path.exists():
    os.system(f"cp {orig_graph_path} {graph_path}")
else:
    print(f"Cannot find the call graph at {orig_graph_path}.")

# Read the profiling results
sio = StringIO()
p = pstats.Stats(str(prof_path), stream=sio)
p.print_stats(str(root).replace("\\", "\\\\"))
# p.print_callers(str(root))
report = sio.getvalue()

# Convert to a DataFrame and save to CSV
report = report.replace(str(root / "*")[:-1], "")  # remove common path
_, report = re.split(r"\n(?= +ncalls)", report, 1)  # remove header
df = (
    pd.read_csv(StringIO(report), sep=" +", engine="python")
    .rename(columns={"percall": "tottime_percall", "percall.1": "cumtime_percall"})
    .sort_values(by="cumtime_percall", ascending=False)
    .query("cumtime > 0.01")
)
df[["filename", "lineno", "function"]] = df.pop(
    "filename:lineno(function)"
).str.extract(r"(.*?):(.*?)\((.*)\)", expand=True)
dt = datetime.datetime.fromtimestamp(prof_path.stat().st_mtime)
df.index = pd.Series([dt.strftime("%Y-%m-%d %H:%M:%S")] * len(df), name="timestamp")
df["label"] = df["filename"].str.extract(r"(\w+).py").squeeze() + "." + df["function"]
report_path = root / "profiling/prof-report.csv"
df.to_csv(report_path, mode="a", header=not report_path.exists())

# Filter and plot the results
df.plot.barh(y=["tottime_percall", "cumtime_percall"], x="label", figsize=(20, 10))
plt.ylabel("")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.legend(loc="lower right")
plt.savefig(root / "profiling/profiling.png")

# Plot benchmark results
df = pd.read_csv(report_path, index_col="label", parse_dates=["timestamp"])
kpis = ["tottime_percall", "cumtime_percall", "tottime", "cumtime"]
labels = df[kpis].idxmax()
bm = df.pivot_table(index="timestamp", columns="label", values=kpis)
bm = bm[list(map(tuple, labels.reset_index().values))]
bm.columns = bm.columns.map(lambda x: f"{x[1]}({x[0]})")
bm.T.plot.barh(figsize=(20, 10))
plt.tight_layout()
plt.legend(loc="lower right")
plt.savefig(root / "profiling/benchmark.png")

# Check performance changes
time_costs = bm.max(axis=1)
t_ratio = time_costs / time_costs.shift(1) - 1
latest_change = t_ratio.iloc[-1]
os.environ["bad_perf"] = 'true'  # for debugging
if latest_change > 1.02:
    os.environ["bad_perf"] = 'true'
