"""
===============================================================
fit_models.py

FCS fitting models for Zeiss LSM980

Models
------
1. Standard 3D diffusion
2. 3D diffusion + Triplet

Returns
-------
tauD
N
G0
Residuals
R²
RMSE
Reduced Chi²
95% confidence intervals

Author : Ram et. al. 2026
===============================================================
"""

import numpy as np
from scipy.optimize import curve_fit


# ============================================================
# Standard 3D diffusion
# ============================================================

def diffusion3D(tau, G0, tauD, S):

    return 1.0 + G0 / (
        (1.0 + tau / tauD)
        * np.sqrt(1.0 + tau / (S * S * tauD))
    )


# ============================================================
# Triplet + 3D diffusion
# ============================================================

def triplet3D(tau, G0, tauD, S, T, tauT):

    triplet = (1.0 - T + T * np.exp(-tau / tauT)) / (1.0 - T)

    diffusion = (
        (1.0 + tau / tauD)
        * np.sqrt(1.0 + tau / (S * S * tauD))
    )

    return 1.0 + G0 * triplet / diffusion


# ============================================================
# Statistics
# ============================================================

def goodness(y, yfit, nparam):

    residual = y - yfit

    ss_res = np.sum(residual ** 2)

    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - ss_res / ss_tot

    rmse = np.sqrt(np.mean(residual ** 2))

    dof = len(y) - nparam

    if dof > 0:
        chi2 = ss_res / dof
    else:
        chi2 = np.nan

    return residual, r2, rmse, chi2


# ============================================================
# Confidence intervals
# ============================================================

def confidence_intervals(popt, pcov):

    sigma = np.sqrt(np.diag(pcov))

    lower = popt - 1.96 * sigma

    upper = popt + 1.96 * sigma

    return sigma, lower, upper


# ============================================================
# Fit Standard Model
# ============================================================

def fit_standard(tau, G, S=7.7):

    tau = np.asarray(tau)

    G = np.asarray(G)

    # -------------------------
    # Initial guess
    # -------------------------

    G0 = max(G) - 1.0

    tauD = tau[np.argmax(G < (1 + G0 / 2))]

    p0 = [G0, tauD]

    # -------------------------
    # Fit
    # -------------------------

    popt, pcov = curve_fit(

        lambda t, G0, tauD:
            diffusion3D(t, G0, tauD, S),

        tau,

        G,

        p0=p0,

        bounds=([0, 1e-8],
                [10, 100]),

        maxfev=10000

    )

    Gfit = diffusion3D(

        tau,

        popt[0],

        popt[1],

        S

    )

    residual, r2, rmse, chi2 = goodness(

        G,

        Gfit,

        2

    )

    sigma, lower, upper = confidence_intervals(

        popt,

        pcov

    )

    result = {

        "model": "Standard 3D",

        "G0": popt[0],

        "tauD": popt[1],

        "N": 1.0 / popt[0],

        "S": S,

        "fit": Gfit,

        "residual": residual,

        "R2": r2,

        "RMSE": rmse,

        "Chi2": chi2,

        "sigma": sigma,

        "lower": lower,

        "upper": upper,

        "covariance": pcov

    }

    return result


# ============================================================
# Fit Triplet Model
# ============================================================

def fit_triplet(tau, G, S=7.7):

    tau = np.asarray(tau)

    G = np.asarray(G)

    G0 = max(G) - 1

    tauD = tau[np.argmax(G < (1 + G0 / 2))]

    p0 = [

        G0,

        tauD,

        0.10,

        5e-6

    ]

    popt, pcov = curve_fit(

        lambda t, G0, tauD, T, tauT:
            triplet3D(

                t,

                G0,

                tauD,

                S,

                T,

                tauT

            ),

        tau,

        G,

        p0=p0,

        bounds=(

            [0, 1e-8, 0, 1e-8],

            [10, 100, 0.8, 1e-2]

        ),

        maxfev=20000

    )

    Gfit = triplet3D(

        tau,

        popt[0],

        popt[1],

        S,

        popt[2],

        popt[3]

    )

    residual, r2, rmse, chi2 = goodness(

        G,

        Gfit,

        4

    )

    sigma, lower, upper = confidence_intervals(

        popt,

        pcov

    )

    result = {

        "model": "Triplet",

        "G0": popt[0],

        "tauD": popt[1],

        "N": 1.0 / popt[0],

        "T": popt[2],

        "tauT": popt[3],

        "S": S,

        "fit": Gfit,

        "residual": residual,

        "R2": r2,

        "RMSE": rmse,

        "Chi2": chi2,

        "sigma": sigma,

        "lower": lower,

        "upper": upper,

        "covariance": pcov

    }

    return result


# ============================================================
# Compare Models
# ============================================================

def compare_models(tau, G, S=7.7):

    std = fit_standard(tau, G, S)

    tri = fit_triplet(tau, G, S)

    if tri["RMSE"] < std["RMSE"]:

        return tri

    return std


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    from read_zeiss import load_zeiss

    datasets = load_zeiss("S1.txt")

    tau = datasets["Average"]["tau"]

    G = datasets["Average"]["G"]

    print()

    print("=========== STANDARD MODEL ===========")

    res = fit_standard(tau, G)

    print(f"N       : {res['N']:.3f}")

    print(f"tauD    : {res['tauD']*1e6:.2f} µs")

    print(f"R²      : {res['R2']:.5f}")

    print(f"RMSE    : {res['RMSE']:.6f}")

    print()

    print("=========== TRIPLET MODEL ===========")

    res = fit_triplet(tau, G)

    print(f"N       : {res['N']:.3f}")

    print(f"tauD    : {res['tauD']*1e6:.2f} µs")

    print(f"T       : {res['T']:.3f}")

    print(f"tauT    : {res['tauT']*1e6:.2f} µs")

    print(f"R²      : {res['R2']:.5f}")

# =====================================================================
# EXTENSIONS: New Biophysical Models & Model Selection (AIC/BIC)
# Append this section to the bottom of your existing fit_models.py
# =====================================================================

# --- NEW MATHEMATICAL MODELS ---

def anomalous3D(tau, G0, tauD, alpha, S):
    """Anomalous 3D Subdiffusion"""
    return 1.0 + G0 / ((1.0 + (tau / tauD)**alpha) * np.sqrt(1.0 + (1.0 / (S * S)) * (tau / tauD)**alpha))

def diffusion2D(tau, G0, tauD):
    """2D Membrane Diffusion"""
    return 1.0 + G0 / (1.0 + tau / tauD)

def triplet2D(tau, G0, tauD, T, tauT):
    """2D Membrane Diffusion with Photophysical Triplet State"""
    triplet = (1.0 - T + T * np.exp(-tau / tauT)) / (1.0 - T)
    return 1.0 + G0 * triplet / (1.0 + tau / tauD)

def triplet3D_2comp(tau, G0, tauD1, tauD2, f1, T, tauT, S):
    """2-Component 3D Diffusion with Photophysical Triplet State"""
    triplet = (1.0 - T + T * np.exp(-tau / tauT)) / (1.0 - T)
    term1 = f1 / ((1.0 + tau / tauD1) * np.sqrt(1.0 + tau / (S * S * tauD1)))
    term2 = (1.0 - f1) / ((1.0 + tau / tauD2) * np.sqrt(1.0 + tau / (S * S * tauD2)))
    return 1.0 + G0 * triplet * (term1 + term2)


# --- STATISTICAL EVALUATION & MODEL SELECTION ---

def compute_aic_bic(y, yfit, nparam):
    """Calculates Akaike (AIC) and Bayesian (BIC) Information Criteria"""
    residual = y - yfit
    ss_res = np.sum(residual ** 2)
    n = len(y)
    
    if ss_res <= 0 or n <= 0:
        return np.nan, np.nan
        
    aic = n * np.log(ss_res / n) + 2 * nparam
    bic = n * np.log(ss_res / n) + nparam * np.log(n)
    return aic, bic