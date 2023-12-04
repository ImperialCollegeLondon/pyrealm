"""Analyze and visualize profiling results."""

import pstats
import re
from io import StringIO
from pathlib import Path

import pandas as pd

# import matplotlib.pyplot as plt

# Read the profiling results
root = Path.cwd()
sio = StringIO()
p = pstats.Stats(str(root / "prof/combined.prof"), stream=sio)
p.print_stats(str(root))
# p.print_callers(str(root))
report = sio.getvalue()

# Convert to a DataFrame and save to CSV
report = re.sub(str(root) + r"[/\\]+", "", report)  # remove directory names
_, report = re.split(r"\n(?= +ncalls)", report, 1)  # remove header
df = (
    pd.read_csv(StringIO(report), sep=" +", engine="python")
    .rename(columns={"percall": "tottime_percall", "percall.1": "cumtime_percall"})
    .sort_values(by="tottime", ascending=False)
)
df[["filename", "lineno", "function"]] = df.pop(
    "filename:lineno(function)"
).str.extract(r"(.*?):(.*?)\((.*)\)", expand=True)
df.to_csv(root / "profiling/prof-report.csv", index=False)

# Filter and plot the results
# df = df.query("tottime > 0.001")
# df["tag"] = df["filename"].str.extract(r"(\w+).py", expand=False) + "."
# + df["function"]
# df.plot.barh(y=["tottime", "cumtime"], x="tag", figsize=(20, 10))
# plt.ylabel("")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.savefig(root / "profiling/prof-report.png")
