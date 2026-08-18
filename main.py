"""
===============================================================
main.py

Complete Zeiss LSM980 FCS Analysis Pipeline

Workflow
--------
1. Read Zeiss correlation file
2. Fit all repetitions
3. Fit average
4. Calculate statistics
5. Generate plots
6. Export results

Author : Ram et. al. 2026
===============================================================
"""

import os

from read_zeiss import load_zeiss
from fit_models import fit_standard, fit_triplet
from statistics import (
    summarize,
    print_summary,
    report_outliers
)

from plotting import (
    create_output_folder,
    plot_all_curves,
    plot_average,
    plot_all_fits,
    plot_tauD,
    plot_histogram,
    plot_boxplot,
    plot_tauD_distribution,
    plot_N_distribution,
    plot_residuals
)

from export_results import export_all


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_FILE = "S1.txt"

MODEL = "standard"
# MODEL = "triplet"

STRUCTURE_PARAMETER = 7.5


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("      Zeiss LSM980 FCS Analysis")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):

        print(f"\nERROR : {INPUT_FILE} not found.\n")
        return

    create_output_folder()

    print("\nReading Zeiss file...")

    datasets = load_zeiss(INPUT_FILE)

    print(f"Loaded {len(datasets)-1} repetitions + Average")

    results = []

    print("\nFitting repetitions...\n")

    for i in range(20):

        name = f"Rep{i+1:02d}"

        tau = datasets[name]["tau"]

        G = datasets[name]["G"]

        if MODEL.lower() == "triplet":

            fit = fit_triplet(
                tau,
                G,
                STRUCTURE_PARAMETER
            )

        else:

            fit = fit_standard(
                tau,
                G,
                STRUCTURE_PARAMETER
            )

        results.append(fit)

        print(
            f"{name:6s}"
            f"   tauD = {fit['tauD']*1e6:8.2f} µs"
            f"   N = {fit['N']:6.2f}"
            f"   R² = {fit['R2']:.5f}"
        )

    print("\nFitting average curve...\n")

    tau = datasets["Average"]["tau"]

    G = datasets["Average"]["G"]

    if MODEL.lower() == "triplet":

        average = fit_triplet(
            tau,
            G,
            STRUCTURE_PARAMETER
        )

    else:

        average = fit_standard(
            tau,
            G,
            STRUCTURE_PARAMETER
        )

    print("=" * 60)

    print("AVERAGE")

    print("=" * 60)

    print(f"Diffusion time : {average['tauD']*1e6:.2f} µs")

    print(f"N              : {average['N']:.3f}")

    print(f"R²             : {average['R2']:.5f}")

    print()

    print("Calculating statistics...\n")

    summary = summarize(results)

    print_summary(summary)

    report_outliers(results)

    print("\nGenerating plots...")

    plot_all_curves(datasets)

    plot_average(
        datasets,
        fit=average["fit"]
    )

    plot_all_fits(
        datasets,
        results
    )

    plot_tauD(results)

    plot_histogram(results)

    plot_boxplot(results)

    plot_tauD_distribution(results)

    plot_N_distribution(results)

    plot_residuals(
        datasets["Average"]["tau"],
        average["residual"]
    )

    print("\nExporting results...")

    export_all(
        results,
        summary
    )

    print()

    print("=" * 60)

    print("Analysis Completed Successfully")

    print("=" * 60)

    print("\nResults saved in current directory.\n")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()
