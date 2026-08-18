"""
#===============================================================
export_results.py

Export FCS fitting results

Outputs
-------
1. Fit_Results.csv
2. Fit_Results.xlsx
3. Summary_Statistics.csv

Author : Ram et. al. 2026
#===============================================================
"""
import pandas as pd


# ============================================================
# Convert results list to DataFrame
# ============================================================

def results_to_dataframe(results):

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

            row["TripletFraction"] = r["T"]
            row["TripletTime_us"] = r["tauT"] * 1e6

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Export repetition results
# ============================================================

def export_results(results,
                   csv_name="Fit_Results.csv",
                   excel_name="Fit_Results.xlsx"):

    df = results_to_dataframe(results)

    df.to_csv(csv_name, index=False)

    try:
        df.to_excel(excel_name, index=False)
        print(f"Saved {excel_name}")
    except Exception:
        print("openpyxl not installed. Excel file skipped.")

    print(f"Saved {csv_name}")

    return df


# ============================================================
# Export summary statistics
# ============================================================

def export_summary(summary,
                   filename="Summary_Statistics.csv"):

    rows = []

    for k, v in summary.items():

        if isinstance(v, tuple):
            value = f"{v[0]} , {v[1]}"
        else:
            value = v

        rows.append([k, value])

    df = pd.DataFrame(rows,
                      columns=["Parameter", "Value"])

    df.to_csv(filename, index=False)

    print(f"Saved {filename}")


# ============================================================
# Export everything
# ============================================================

def export_all(results,
               summary):

    export_results(results)

    export_summary(summary)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    from read_zeiss import load_zeiss
    from fit_models import fit_standard
    from statistics import summarize

    datasets = load_zeiss("S1.txt")

    results = []

    for i in range(20):

        name = f"Rep{i+1:02d}"

        tau = datasets[name]["tau"]
        G = datasets[name]["G"]

        results.append(
            fit_standard(tau, G)
        )

    summary = summarize(results)

    export_all(results, summary)

    print("Export completed successfully.")
