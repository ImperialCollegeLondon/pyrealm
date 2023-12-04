"""Analyze and visualize profiling results."""

import datetime
import pstats
import re
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Read the profiling results
root = Path.cwd()
sio = StringIO()
prof_path = root / "profiling/profiling.prof"
p = pstats.Stats(str(prof_path), stream=sio)
p.print_stats(str(root))
# p.print_callers(str(root))
report = sio.getvalue()

# Convert to a DataFrame and save to CSV
report = re.sub(str(root) + r"[/\\]+", "", report)  # remove directory names
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
