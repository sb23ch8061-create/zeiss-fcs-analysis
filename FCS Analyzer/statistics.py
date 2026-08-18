"""
===============================================================
statistics.py

Statistical analysis of FCS fitting results

Calculates
----------
Mean
Median
Standard deviation
Standard error
Coefficient of variation
95% confidence interval
Outlier detection (IQR)
Summary table

Author : Ram et. al. 2026
===============================================================
"""

import numpy as np
import pandas as pd


# ============================================================
# Mean
# ============================================================

def mean(x):
    return np.mean(np.asarray(x))


# ============================================================
# Median
# ============================================================

def median(x):
    return np.median(np.asarray(x))


# ============================================================
# Standard deviation
# ============================================================

def std(x):

    return np.std(np.asarray(x), ddof=1)


# ============================================================
# Standard error
# ============================================================

def sem(x):

    x = np.asarray(x)

    return std(x) / np.sqrt(len(x))


# ============================================================
# Coefficient of variation
# ============================================================

def cv(x):

    x = np.asarray(x)

    return 100 * std(x) / mean(x)


# ============================================================
# 95% confidence interval
# ============================================================

def confidence95(x):

    x = np.asarray(x)

    m = mean(x)

    s = sem(x)

    return (

        m - 1.96 * s,

        m + 1.96 * s

    )


# ============================================================
# IQR outlier detection
# ============================================================

def detect_outliers(x):

    x = np.asarray(x)

    q1 = np.percentile(x, 25)

    q3 = np.percentile(x, 75)

    iqr = q3 - q1

    low = q1 - 1.5 * iqr

    high = q3 + 1.5 * iqr

    mask = (x < low) | (x > high)

    return mask


# ============================================================
# Build dataframe from fitting results
# ============================================================

def results_dataframe(results):

    rows = []

    for i, r in enumerate(results):

        row = {

            "Repetition": i + 1,

            "Model": r["model"],

            "tauD_s": r["tauD"],

            "tauD_us": r["tauD"] * 1e6,

            "N": r["N"],

            "G0": r["G0"],

            "R2": r["R2"],

            "RMSE": r["RMSE"],

            "Chi2": r["Chi2"]

        }

        if "T" in r:

            row["Triplet"] = r["T"]

            row["tauTriplet_us"] = r["tauT"] * 1e6

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Summary statistics
# ============================================================

def summarize(results):

    df = results_dataframe(results)

    tau = df["tauD_us"].values

    N = df["N"].values

    R2 = df["R2"].values

    summary = {

        "Number of repetitions": len(df),

        "tauD_mean_us": mean(tau),

        "tauD_median_us": median(tau),

        "tauD_sd_us": std(tau),

        "tauD_sem_us": sem(tau),

        "tauD_cv_percent": cv(tau),

        "tauD_CI95": confidence95(tau),

        "N_mean": mean(N),

        "N_sd": std(N),

        "N_sem": sem(N),

        "N_cv_percent": cv(N),

        "N_CI95": confidence95(N),

        "Mean_R2": mean(R2)

    }

    return summary


# ============================================================
# Pretty print
# ============================================================

def print_summary(summary):

    print()

    print("====================================")
    print("FCS STATISTICS")
    print("====================================")

    print(f"Repetitions      : {summary['Number of repetitions']}")

    print()

    print("Diffusion time")

    print("---------------------------")

    print(f"Mean      : {summary['tauD_mean_us']:.3f} µs")

    print(f"Median    : {summary['tauD_median_us']:.3f} µs")

    print(f"SD        : {summary['tauD_sd_us']:.3f} µs")

    print(f"SEM       : {summary['tauD_sem_us']:.3f} µs")

    print(f"CV        : {summary['tauD_cv_percent']:.2f} %")

    ci = summary["tauD_CI95"]

    print(f"95% CI    : {ci[0]:.3f} - {ci[1]:.3f} µs")

    print()

    print("Number of molecules")

    print("---------------------------")

    print(f"Mean N    : {summary['N_mean']:.3f}")

    print(f"SD        : {summary['N_sd']:.3f}")

    print(f"SEM       : {summary['N_sem']:.3f}")

    print(f"CV        : {summary['N_cv_percent']:.2f} %")

    ci = summary["N_CI95"]

    print(f"95% CI    : {ci[0]:.3f} - {ci[1]:.3f}")

    print()

    print(f"Mean R²   : {summary['Mean_R2']:.5f}")

    print("====================================")


# ============================================================
# Outlier report
# ============================================================

def report_outliers(results):

    df = results_dataframe(results)

    mask = detect_outliers(df["tauD_us"].values)

    idx = np.where(mask)[0]

    print()

    print("Outlier report")

    print("----------------")

    if len(idx) == 0:

        print("No outliers detected.")

        return

    for i in idx:

        print(

            f"Rep {i+1:02d} : "

            f"{df['tauD_us'][i]:.3f} µs"

        )


# ============================================================
# Save statistics
# ============================================================

def save_summary(summary, filename="Statistics.csv"):

    rows = []

    for k, v in summary.items():

        rows.append([k, v])

    df = pd.DataFrame(rows, columns=["Parameter", "Value"])

    df.to_csv(filename, index=False)

    print(f"Saved {filename}")


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    from read_zeiss import load_zeiss
    from fit_models import fit_standard

    datasets = load_zeiss("S1.txt")

    results = []

    # Fit first 20 repetitions
    for i in range(20):

        name = f"Rep{i+1:02d}"

        tau = datasets[name]["tau"]

        G = datasets[name]["G"]

        res = fit_standard(tau, G)

        results.append(res)

    df = results_dataframe(results)

    print(df.head())

    summary = summarize(results)

    print_summary(summary)

    report_outliers(results)

    save_summary(summary)
