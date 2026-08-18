"""
===============================================================
plotting.py (Part 1)

Publication-quality plotting functions for FCS

Author : Ram et. al. 2026

===============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# Create output folder
# -------------------------------------------------------------

def create_output_folder(folder="Results"):

    if not os.path.exists(folder):
        os.makedirs(folder)

    return folder


# -------------------------------------------------------------
# Global plotting style
# -------------------------------------------------------------

def set_plot_style():

    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6
    plt.rcParams["figure.dpi"] = 120


# -------------------------------------------------------------
# Plot one FCS curve
# -------------------------------------------------------------

def plot_single_curve(
        tau,
        G,
        fit=None,
        title="FCS",
        save=None
):

    set_plot_style()

    plt.figure(figsize=(6,5))

    plt.semilogx(
        tau,
        G,
        "o",
        ms=4,
        label="Experiment"
    )

    if fit is not None:

        plt.semilogx(
            tau,
            fit,
            "-",
            lw=2,
            label="Fit"
        )

    plt.xlabel("Lag time (s)")

    plt.ylabel("G($\\tau$)")

    plt.title(title)

    plt.legend()

    plt.tight_layout()

    if save is not None:

        plt.savefig(save, dpi=300)

    plt.show()


# -------------------------------------------------------------
# Plot all repetitions
# -------------------------------------------------------------

def plot_all_curves(
        datasets,
        save="Results/all_curves.png"
):

    create_output_folder()

    set_plot_style()

    plt.figure(figsize=(7,5))

    names = sorted(datasets.keys())

    for name in names:

        if name == "Average":

            continue

        tau = datasets[name]["tau"]

        G = datasets[name]["G"]

        plt.semilogx(

            tau,

            G,

            lw=1

        )

    plt.xlabel("Lag time (s)")

    plt.ylabel("G($\\tau$)")

    plt.title("All repetitions")

    plt.tight_layout()

    plt.savefig(save, dpi=300)

    plt.show()


# -------------------------------------------------------------
# Plot average curve
# -------------------------------------------------------------

def plot_average(
        datasets,
        fit=None,
        save="Results/average_fit.png"
):

    create_output_folder()

    set_plot_style()

    tau = datasets["Average"]["tau"]

    G = datasets["Average"]["G"]

    plt.figure(figsize=(6,5))

    plt.semilogx(

        tau,

        G,

        "o",

        ms=4,

        label="Average"

    )

    if fit is not None:

        plt.semilogx(

            tau,

            fit,

            "-",

            lw=2,

            label="Fit"

        )

    plt.xlabel("Lag time (s)")

    plt.ylabel("G($\\tau$)")

    plt.legend()

    plt.tight_layout()

    plt.savefig(save, dpi=300)

    plt.show()


# -------------------------------------------------------------
# Overlay all fitted curves
# -------------------------------------------------------------

def plot_all_fits(
        datasets,
        results,
        save="Results/all_fits.png"
):

    create_output_folder()

    set_plot_style()

    plt.figure(figsize=(7,5))

    for i in range(20):

        name = f"Rep{i+1:02d}"

        tau = datasets[name]["tau"]

        G = datasets[name]["G"]

        plt.semilogx(

            tau,

            G,

            color="gray",

            alpha=0.35

        )

        plt.semilogx(

            tau,

            results[i]["fit"],

            lw=2

        )

    plt.xlabel("Lag time (s)")

    plt.ylabel("G($\\tau$)")

    plt.title("All fitted curves")

    plt.tight_layout()

    plt.savefig(save, dpi=300)

    plt.show()


# -------------------------------------------------------------
# Error-bar plot of diffusion times
# -------------------------------------------------------------

def plot_tauD(
        results,
        save="Results/tauD.png"
):

    create_output_folder()

    set_plot_style()

    tau = []

    err = []

    rep = []

    for i, r in enumerate(results):

        tau.append(

            r["tauD"]*1e6

        )

        err.append(

            r["sigma"][1]*1e6

        )

        rep.append(i+1)

    plt.figure(figsize=(8,4))

    plt.errorbar(

        rep,

        tau,

        yerr=err,

        fmt="o",

        capsize=3

    )

    plt.xlabel("Repetition")

    plt.ylabel("Diffusion time (µs)")

    plt.tight_layout()

    plt.savefig(save, dpi=300)

    plt.show()


# -------------------------------------------------------------
# Compare standard vs triplet fit
# -------------------------------------------------------------

def compare_models_plot(
        tau,
        G,
        fit1,
        fit2,
        save="Results/model_comparison.png"
):

    create_output_folder()

    set_plot_style()

    plt.figure(figsize=(6,5))

    plt.semilogx(

        tau,

        G,

        "o",

        ms=4,

        label="Experiment"

    )

    plt.semilogx(

        tau,

        fit1,

        lw=2,

        label="Standard"

    )

    plt.semilogx(

        tau,

        fit2,

        lw=2,

        label="Triplet"

    )

    plt.xlabel("Lag time (s)")

    plt.ylabel("G($\\tau$)")

    plt.legend()

    plt.tight_layout()

    plt.savefig(save, dpi=300)

    plt.show()

# =============================================================
# plotting.py (Part 2)
# Append this below Part 1
# =============================================================

# -------------------------------------------------------------
# Histogram of diffusion times
# -------------------------------------------------------------

def plot_histogram(
        results,
        save="Results/tauD_histogram.png"
):

    create_output_folder()

    set_plot_style()

    tau = [r["tauD"]*1e6 for r in results]

    plt.figure(figsize=(6,5))

    plt.hist(
        tau,
        bins="auto",
        edgecolor="black"
    )

    plt.xlabel("Diffusion time (µs)")
    plt.ylabel("Frequency")
    plt.title("Distribution of diffusion time")

    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.show()


# -------------------------------------------------------------
# Box plot
# -------------------------------------------------------------

def plot_boxplot(
        results,
        save="Results/tauD_boxplot.png"
):

    create_output_folder()

    set_plot_style()

    tau = [r["tauD"]*1e6 for r in results]

    plt.figure(figsize=(4,6))

    plt.boxplot(
        tau,
        vert=True,
        patch_artist=True
    )

    plt.ylabel("Diffusion time (µs)")
    plt.title("Boxplot")

    plt.tight_layout()
    plt.savefig(save, dpi=300)
    plt.show()


# -------------------------------------------------------------
# Residual plot
# -------------------------------------------------------------

def plot_residuals(
        tau,
        residual,
        save="Results/residuals.png"
):

    create_output_folder()

    set_plot_style()

    plt.figure(figsize=(6,4))

    plt.semilogx(
        tau,
        residual,
        "o",
        ms=3
    )

    plt.axhline(
        0,
        color="k",
        ls="--"
    )

    plt.xlabel("Lag time (s)")
    plt.ylabel("Residual")

    plt.tight_layout()

    plt.savefig(save, dpi=300)

    plt.show()


# -------------------------------------------------------------
# Distribution of N
# -------------------------------------------------------------

def plot_N_distribution(
        results,
        save="Results/N_histogram.png"
):

    create_output_folder()

    set_plot_style()

    N = [r["N"] for r in results]

    plt.figure(figsize=(6,5))

    plt.hist(
        N,
        bins="auto",
        edgecolor="black"
    )

    plt.xlabel("Number of molecules (N)")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(save, dpi=300)

    plt.show()


# -------------------------------------------------------------
# tauD distribution
# -------------------------------------------------------------

def plot_tauD_distribution(
        results,
        save="Results/tauD_distribution.png"
):

    create_output_folder()

    set_plot_style()

    tau = np.array(
        [r["tauD"]*1e6 for r in results]
    )

    plt.figure(figsize=(7,4))

    plt.plot(
        np.arange(1,len(tau)+1),
        tau,
        "o-"
    )

    plt.xlabel("Repetition")

    plt.ylabel("Diffusion time (µs)")

    plt.tight_layout()

    plt.savefig(save,dpi=300)

    plt.show()


# -------------------------------------------------------------
# Dashboard
# -------------------------------------------------------------

def plot_summary_dashboard(
        results,
        datasets
):

    plot_all_curves(datasets)

    plot_average(
        datasets,
        fit=results[-1]["fit"]
    )

    plot_histogram(results)

    plot_boxplot(results)

    plot_tauD(results)

    plot_N_distribution(results)


# -------------------------------------------------------------
# Test
# -------------------------------------------------------------

if __name__ == "__main__":

    from read_zeiss import load_zeiss
    from fit_models import fit_standard

    datasets = load_zeiss("S1.txt")

    results = []

    # Fit all repetitions

    for i in range(20):

        name = f"Rep{i+1:02d}"

        tau = datasets[name]["tau"]

        G = datasets[name]["G"]

        res = fit_standard(
            tau,
            G
        )

        results.append(res)

    # Fit average

    avg = fit_standard(
        datasets["Average"]["tau"],
        datasets["Average"]["G"]
    )

    results.append(avg)

    plot_summary_dashboard(
        results[:-1],
        datasets
    )

    plot_residuals(
        datasets["Average"]["tau"],
        avg["residual"]
    )

    print("All plots generated successfully.")
