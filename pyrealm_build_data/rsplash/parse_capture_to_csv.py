"""Python script to pass rsplash display() outputs to a CSV file."""

from itertools import batched

import pandas as pd

with open("capture.out") as cap:
    rsplash_capture = [line.strip() for line in cap.readlines()]

start = rsplash_capture.index("SOLAR variable list:")
rsplash_capture = rsplash_capture[start:]
end = rsplash_capture.index(">")
rsplash_capture = rsplash_capture[:end]

# Work out the record length
rec_length = rsplash_capture.index("SOLAR variable list:", 1)
# Remove the EVAP and SOLAR headers
rsplash_capture = [line for line in rsplash_capture if "variable list" not in line]

# Split into daily records
records = []

for record in batched(rsplash_capture, rec_length - 2):
    # Take a list of strings like "PPFD: 12.211241 mol/m^2", split the strings on spaces
    # and then build a dictionary of variable name and float values, discarding units.
    parts = [var.split(" ") for var in record]
    parts_dict = {var[0][:-1]: float(var[1]) for var in parts}
    records.append(parts_dict)

df = pd.DataFrame.from_records(records)

# Add dates from input file
inputs = pd.read_csv("rsplash_Bourne_inputs.csv")
df.insert(0, "date", inputs.date)

df.to_csv("rsplash_Bourne_internals.csv")
