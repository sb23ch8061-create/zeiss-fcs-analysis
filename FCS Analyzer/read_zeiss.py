"""
=========================================================
read_zeiss.py

Read Zeiss LSM980 FCS correlation export or file stream

Expected format
---------------
42 columns (or 2*N columns): Tau1 G1 Tau2 G2 ... TauN GN TauAvg GAvg

Author : Ram et. al. 2026
=========================================================
"""

import numpy as np


class ZeissFCS:

    def __init__(self, filename):

        self.filename = filename
        self.data = None
        self.datasets = {}

    # ----------------------------------------------------
    # Read file (Handles both file paths and Streamlit uploads)
    # ----------------------------------------------------

    def read(self):

        rows = []

        # Determine file source type
        if isinstance(self.filename, (str, bytes)):
            with open(self.filename, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        elif hasattr(self.filename, "getvalue"):
            raw_data = self.filename.getvalue()
            if isinstance(raw_data, bytes):
                raw_data = raw_data.decode("utf-8", errors="ignore")
            lines = raw_data.splitlines()
        elif hasattr(self.filename, "read"):
            content = self.filename.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            lines = content.splitlines()
        else:
            lines = self.filename

        for line in lines:

            line = line.strip()

            if line == "":
                continue

            try:
                values = [float(x) for x in line.split()]
                rows.append(values)

            except ValueError:
                # Ignore header/non-numeric lines
                continue

        if len(rows) == 0:
            raise ValueError("No numeric data found in file.")

        # -------------------------------------------------
        # Determine correct number of columns
        # -------------------------------------------------

        lengths = [len(r) for r in rows]

        ncol = max(set(lengths), key=lengths.count)

        rows = [r for r in rows if len(r) == ncol]

        self.data = np.asarray(rows, dtype=float)

        if self.data.shape[1] % 2 != 0:
            raise ValueError(
                f"Expected an even number of columns, got {self.data.shape[1]}"
            )

        return self.data

    # ----------------------------------------------------
    # Number of datasets
    # ----------------------------------------------------

    def number_of_datasets(self):

        return self.data.shape[1] // 2

    # ----------------------------------------------------
    # Extract all repetitions
    # ----------------------------------------------------

    def extract(self):

        self.datasets = {}

        ndatasets = self.number_of_datasets()

        for i in range(ndatasets):

            tau = self.data[:, 2 * i]
            G = self.data[:, 2 * i + 1]

            mask = np.isfinite(tau) & np.isfinite(G)

            tau = tau[mask]
            G = G[mask]

            if i == ndatasets - 1:
                name = "Average"
            else:
                name = f"Rep{i+1:02d}"

            self.datasets[name] = {
                "tau": tau,
                "G": G
            }

        return self.datasets

    # ----------------------------------------------------
    # Print summary
    # ----------------------------------------------------

    def summary(self):

        print("\n========== DATA SUMMARY ==========\n")

        print(f"Datasets : {len(self.datasets)}\n")

        for name in self.datasets:

            n = len(self.datasets[name]["tau"])

            print(f"{name:10s} {n:5d} points")

        print()

    # ----------------------------------------------------
    # Dataset names
    # ----------------------------------------------------

    def names(self):

        return list(self.datasets.keys())

    # ----------------------------------------------------
    # Get one dataset
    # ----------------------------------------------------

    def get(self, name):

        return self.datasets[name]


# ==========================================================
# Convenience function
# ==========================================================

def load_zeiss(filename):

    reader = ZeissFCS(filename)

    reader.read()

    reader.extract()

    return reader.datasets