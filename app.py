import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import scipy.constants as const
from pathlib import Path
from read_zeiss import load_zeiss
from fit_models import fit_standard, fit_triplet
from export_results import results_to_dataframe
from auth_db import (
    verify_user, 
    create_user, 
    create_project, 
    get_user_projects, 
    save_dataset_record, 
    get_project_datasets
)

# Wrapper inside app.py to preserve original fit_models.py untouched
def fit_fcs_model(tau, G, model_type="Standard 3D", S=7.5):
    tau = np.asarray(tau)
    G = np.asarray(G)
    n = len(G)

    if model_type == "Triplet 3D":
        res = fit_triplet(tau, G, S=S)
        nparam = 4
    else:
        res = fit_standard(tau, G, S=S)
        nparam = 2

    Gfit = res.get("fit", G)
    residual = G - Gfit
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((G - np.mean(G)) ** 2)

    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    chi2 = ss_res / (n - nparam) if (n - nparam) > 0 else ss_res
    
    aic = n * np.log(ss_res / n) + 2 * nparam if ss_res > 0 else 0.0
    bic = n * np.log(ss_res / n) + nparam * np.log(n) if ss_res > 0 else 0.0

    res["R2"] = res.get("R2", r2)
    res["Chi2"] = res.get("Chi2", chi2)
    res["AIC"] = aic
    res["BIC"] = bic
    res["residual"] = residual
    if "N" not in res and "G0" in res:
        res["N"] = 1.0 / res["G0"] if res["G0"] > 0 else 0.0

    return res

# Page Configuration
st.set_page_config(page_title="Zeiss LSM980 FCS Suite", layout="wide", page_icon="🔬")

# Custom CSS for Copyright Watermark Footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #808495;
        text-align: center;
        padding: 8px 0px;
        font-size: 13px;
        border-top: 1px solid #262730;
        z-index: 999;
    }
    .main .block-container {
        padding-bottom: 60px;
    }
    </style>
""", unsafe_allow_html=True)

# Session State Initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "current_project" not in st.session_state:
    st.session_state.current_project = None
if "user_data_store" not in st.session_state:
    st.session_state.user_data_store = {}
if "omega_sq" not in st.session_state:
    st.session_state.omega_sq = 0.042

# --- USER AUTHENTICATION MODULE ---
def login_screen():
    st.title("🔬 Zeiss LSM980 FCS Real-Time Analysis Suite")
    st.subheader("Database-Backed User Authentication & Data Workspace")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Register Account"])
        
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username / Lab ID")
                password = st.text_input("Password", type="password")
                submit_btn = st.form_submit_button("Enter Analysis Suite", type="primary")
                
                if submit_btn:
                    if username.strip() == "" or password.strip() == "":
                        st.error("Please enter both username and password.")
                    else:
                        if verify_user(username.strip(), password):
                            st.session_state.authenticated = True
                            st.session_state.username = username.strip()
                            st.success("Authentication successful!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")
                            
        with tab_register:
            with st.form("register_form"):
                new_username = st.text_input("Choose Username / Lab ID")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                reg_btn = st.form_submit_button("Create Account")
                
                if reg_btn:
                    if new_username.strip() == "" or new_password.strip() == "":
                        st.error("All fields are required.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        success, msg = create_user(new_username.strip(), new_password)
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                
    with col2:
        st.info("""
        **Platform Features:**
        * Multi-Model FCS Fitting (Standard 3D, Triplet 3D).
        * Automated Model Selection (AIC & BIC criteria).
        * Stokes-Einstein Hydrodynamic Radius ($R_h$) & Molar Concentration ($C$) Calculators.
        * Secure SQLite-backed user authentication and isolated data storage.
        """)

# --- AUTHENTICATION GATE ---
if not st.session_state.authenticated:
    login_screen()
    st.markdown('<div class="footer">© 2026 Zeiss LSM980 FCS Analysis Suite | Developed by SHIBASISH | Confidential & Proprietary</div>', unsafe_allow_html=True)
    st.stop()


# =====================================================================
# --- MAIN APPLICATION WORKSPACE ---
# =====================================================================
st.sidebar.markdown(f"👤 **User:** `{st.session_state.username}`")
if st.sidebar.button("Log Out"):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.current_project = None
    st.rerun()

st.title("Zeiss LSM980 FCS Real-Time Dashboard")

# --- PROJECT WORKSPACE UI ---
st.subheader("📁 Project Workspace")
project_col1, project_col2 = st.columns([1, 2])

user_projects = get_user_projects(st.session_state.username)
project_names = [p["name"] for p in user_projects]

with project_col1:
    with st.form("new_project_form", clear_on_submit=True):
        new_proj_name = st.text_input("Create New Project", placeholder="e.g., Cell Line A - Temp Exp")
        create_proj_btn = st.form_submit_button("Create")
        if create_proj_btn:
            if new_proj_name:
                success, msg = create_project(st.session_state.username, new_proj_name)
                if success:
                    st.success(msg)
                    st.session_state.current_project = new_proj_name
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.error("Please enter a project name.")

with project_col2:
    if project_names:
        default_idx = project_names.index(st.session_state.current_project) if st.session_state.current_project in project_names else 0
        
        selected_project = st.selectbox(
            "Active Project", 
            ["-- Select a Project --"] + project_names,
            index=(default_idx + 1) if st.session_state.current_project else 0
        )
        
        if selected_project != "-- Select a Project --":
            if st.session_state.current_project != selected_project:
                st.session_state.current_project = selected_project
                st.rerun()
        else:
            st.session_state.current_project = None
    else:
        st.info("You don't have any projects yet. Create one to get started!")
        st.session_state.current_project = None

st.divider()

# --- ANALYSIS WORKSPACE (PROTECTED & CONDITIONAL) ---
if st.session_state.current_project:
    st.markdown(f"### Current Workspace: **{st.session_state.current_project}**")
    
    # Ensure project directory exists
    project_dir = Path("user_data") / st.session_state.username / st.session_state.current_project
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Load saved datasets for this project
    saved_datasets = get_project_datasets(st.session_state.username, st.session_state.current_project)
    
    source_tab1, source_tab2 = st.tabs(["📤 Upload New File", "📂 Saved Project Datasets"])
    
    active_file_stream = None
    active_filename = None
    
    with source_tab1:
        uploaded_file = st.file_uploader(
            "📂 Upload Zeiss FCS Raw Data File (.txt, .dat, .csv)", 
            type=["txt", "dat", "csv"],
            key="uploader"
        )
        if uploaded_file is not None:
            # Save uploaded file to project directory
            saved_path = project_dir / uploaded_file.name
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Save record to DB
            save_dataset_record(st.session_state.username, st.session_state.current_project, uploaded_file.name, str(saved_path))
            
            active_file_stream = saved_path
            active_filename = uploaded_file.name
            st.success(f"File **{uploaded_file.name}** saved to project folder.")

    with source_tab2:
        if saved_datasets:
            dataset_options = {d["filename"]: Path(d["file_path"]) for d in saved_datasets}
            selected_saved = st.selectbox("Select a previously uploaded dataset:", list(dataset_options.keys()))
            if selected_saved:
                active_file_stream = dataset_options[selected_saved]
                active_filename = selected_saved
        else:
            st.info("No saved datasets found in this project yet.")

    @st.cache_data
    def get_data(file_path):
        try:
            # We open the saved file and wrap it in BytesIO to perfectly mimic
            # Streamlit's original virtual file behavior for the Zeiss reader.
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            return load_zeiss(io.BytesIO(file_bytes))
        except Exception as e:
            st.error(f"Error reading file content: {e}")
            return None

    if active_file_stream is not None:
        datasets = get_data(active_file_stream)

        if datasets:
            rep_keys = [k for k in datasets.keys() if k != "Average"]

            # --- SIDEBAR CONTROLS ---
            st.sidebar.header("⚙️ Model & Fit Parameters")
            
            s_value = st.sidebar.slider("Structure Parameter (S)", min_value=1.0, max_value=15.0, value=7.5, step=0.1)
            
            model_options = ["Standard 3D", "Triplet 3D"]
            model_choice = st.sidebar.selectbox("Select FCS Model", model_options)
            
            selected_reps = st.sidebar.multiselect(
                "Select Repetitions to Compare", 
                options=rep_keys, 
                default=rep_keys[:2] if len(rep_keys) >= 2 else rep_keys
            )

            st.sidebar.divider()
            st.sidebar.subheader("🛡️ Data Quality Controls")
            min_r2 = st.sidebar.slider("Minimum R² Threshold Filter", 0.80, 0.999, 0.90, 0.005)

            # --- PHYSICAL PARAMETERS & CALCULATORS ---
            st.sidebar.divider()
            st.sidebar.subheader("📐 Physical Calculators")
            
            with st.sidebar.expander("Confocal Parameter Calculation (ω²)"):
                d_val = st.number_input("Diffusivity (D)", value=400.0, format="%.6f")
                d_unit = st.selectbox("Unit for D", ["µm²/s", "cm²/s"])
                tau_val = st.number_input("Diffusion Time (τD)", value=25.0, format="%.6f")
                tau_unit = st.selectbox("Unit for τD", ["µs", "ms", "s"])
                
                if st.button("Calculate ω²"):
                    d_um2 = d_val if d_unit == "µm²/s" else d_val * 1e8
                    t_s = tau_val * 1e-6 if tau_unit == "µs" else (tau_val * 1e-3 if tau_unit == "ms" else tau_val)
                    st.session_state.omega_sq = 4 * d_um2 * t_s
                    
                st.success(f"Current ω²: {st.session_state.omega_sq:.6e} µm²")

            with st.sidebar.expander("Hydrodynamic Radius (Rh) & Conc."):
                temp_c = st.number_input("Temperature (°C)", value=25.0)
                viscosity_mPas = st.number_input("Solvent Viscosity η (mPa·s)", value=0.890)
                
                temp_k = temp_c + 273.15
                viscosity_pas = viscosity_mPas * 1e-3

            # --- 2. FIT PROCESSING ENGINE ---
            results = {}

            with st.spinner(f"Fitting data from '{active_filename}' using model [{model_choice}]..."):
                for name in rep_keys:
                    tau = np.array(datasets[name]["tau"])
                    G = np.array(datasets[name]["G"])
                    
                    mask = tau >= 1e-6
                    res = fit_fcs_model(tau[mask], G[mask], model_type=model_choice, S=s_value)
                    
                    if res["R2"] >= min_r2:
                        results[name] = res

                avg_tau = np.array(datasets["Average"]["tau"])
                avg_G = np.array(datasets["Average"]["G"])
                avg_mask = avg_tau >= 1e-6
                avg_res = fit_fcs_model(avg_tau[avg_mask], avg_G[avg_mask], model_type=model_choice, S=s_value)

            # --- 3. GRAPHICAL PLOTS SIDE-BY-SIDE ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Average Curve Fit & Residuals")
                fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                
                ax1.semilogx(avg_tau, avg_G, "o", ms=4, label="Experiment (Avg)", color="gray", alpha=0.3)
                ax1.semilogx(avg_tau[avg_mask], avg_res["fit"], "-", lw=2, label=f"Fit ({model_choice})", color="red")
                ax1.set_ylabel("G(tau)")
                ax1.legend(fontsize="small")
                ax1.grid(True, which="both", ls="--", alpha=0.3)
                
                residuals = avg_G[avg_mask] - avg_res["fit"]
                ax2.semilogx(avg_tau[avg_mask], residuals, "-", color="blue", lw=1.2)
                ax2.axhline(0, color="black", linestyle="--", alpha=0.7)
                ax2.set_xlabel("Lag time (s)")
                ax2.set_ylabel("Residuals")
                ax2.grid(True, which="both", ls="--", alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig1)
                
                buf1 = io.BytesIO()
                fig1.savefig(buf1, format="png", dpi=300, bbox_inches='tight')
                buf1.seek(0)
                
            with col2:
                st.subheader("Selected Repetition Fit & Residuals")
                buf2 = None
                if selected_reps:
                    fig2, (rax1, rax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
                    
                    for rep_name in selected_reps:
                        if rep_name in results:
                            rep_tau = np.array(datasets[rep_name]["tau"])
                            rep_G = np.array(datasets[rep_name]["G"])
                            rep_mask = rep_tau >= 1e-6
                            rep_res = results[rep_name]
                            
                            line_match = rax1.semilogx(rep_tau, rep_G, "o", ms=3, alpha=0.25)
                            active_color = line_match[0].get_color()
                            
                            rax1.semilogx(rep_tau[rep_mask], rep_res["fit"], "-", lw=1.5, color=active_color, label=f"{rep_name}")
                            rax2.semilogx(rep_tau[rep_mask], rep_G[rep_mask] - rep_res["fit"], "-", lw=1.2, color=active_color)
                    
                    rax1.set_ylabel("G(tau)")
                    rax1.legend(fontsize='small', loc='upper right')
                    rax1.grid(True, which="both", ls="--", alpha=0.3)
                    rax2.axhline(0, color="black", linestyle="--", alpha=0.7)
                    rax2.set_xlabel("Lag time (s)")
                    rax2.set_ylabel("Residuals")
                    rax2.grid(True, which="both", ls="--", alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
                    buf2 = io.BytesIO()
                    fig2.savefig(buf2, format="png", dpi=300, bbox_inches='tight')
                    buf2.seek(0)

            st.divider()

            # --- 4. SUMMARY STATISTICS & DERIVED PHYSICAL PARAMETERS ---
            st.subheader("Summary Statistics & Physical Biophysics Parameters")
            
            if results:
                tauD_sec_list = np.array([r["tauD"] for r in results.values()])
                tauD_us_list = tauD_sec_list * 1e6
                D_list_um2s = st.session_state.omega_sq / (4.0 * tauD_sec_list)
                D_list_m2s = D_list_um2s * 1e-12

                Rh_nm_list = (const.k * temp_k) / (6.0 * np.pi * viscosity_pas * D_list_m2s) * 1e9

                w0_um = np.sqrt(st.session_state.omega_sq)
                V_eff_L = (np.pi ** 1.5) * s_value * (w0_um ** 3) * 1e-15
                
                N_list = np.array([r["N"] for r in results.values()])
                Conc_nM_list = (N_list / (V_eff_L * const.Avogadro)) * 1e9

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Diffusion Time (τD)", f"{np.mean(tauD_us_list):.2f} ± {np.std(tauD_us_list):.2f} µs")
                m2.metric("Diffusivity (D)", f"{np.mean(D_list_um2s):.2f} ± {np.std(D_list_um2s):.2f} µm²/s")
                m3.metric("Hydrodynamic Radius (Rh)", f"{np.mean(Rh_nm_list):.3f} ± {np.std(Rh_nm_list):.3f} nm")
                m4.metric("Molar Concentration (C)", f"{np.mean(Conc_nM_list):.3f} ± {np.std(Conc_nM_list):.3f} nM")

                df_results = results_to_dataframe(list(results.values()))
                df_results["Diffusivity D (µm²/s)"] = D_list_um2s
                df_results["Rh (nm)"] = Rh_nm_list
                df_results["Concentration (nM)"] = Conc_nM_list

                st.dataframe(df_results, height=350, use_container_width=True)

            st.subheader("Model Selection & Information Criteria (Average Dataset)")
            aic_col, bic_col, chi_col, r2_col = st.columns(4)
            aic_col.metric("Akaike Info Criterion (AIC)", f"{avg_res['AIC']:.2f}")
            bic_col.metric("Bayesian Info Criterion (BIC)", f"{avg_res['BIC']:.2f}")
            chi_col.metric("Reduced Chi²", f"{avg_res['Chi2']:.6f}")
            r2_col.metric("Coefficient of Determination (R²)", f"{avg_res['R2']:.5f}")

            st.divider()

            # --- 5. DATA PERSISTENCE & EXPORT SECTION ---
            st.subheader("💾 Save & Export Session Analysis")
            save_prefix = st.text_input("Filename / Session Name Prefix:", value=f"{active_filename}_fitted" if active_filename else "FCS_Analysis")

            dl1, dl2, dl3, dl4 = st.columns(4)
            with dl1:
                if 'df_results' in locals():
                    st.download_button("📊 Download Full CSV Data", data=df_results.to_csv(index=False).encode('utf-8'), file_name=f"{save_prefix}_stats.csv", mime='text/csv')
            with dl2:
                st.download_button("📈 Download Average Graph", data=buf1, file_name=f"{save_prefix}_AvgFit.png", mime='image/png')
            with dl3:
                if buf2:
                    st.download_button("📉 Download Repetition Graph", data=buf2, file_name=f"{save_prefix}_RepsFit.png", mime='image/png')
            with dl4:
                if st.button("💾 Save Session to Account"):
                    if 'df_results' in locals():
                        st.session_state.user_data_store[save_prefix] = df_results
                        st.success(f"Saved session '{save_prefix}' to user account '{st.session_state.username}'.")

    else:
        st.info("👆 Please upload a new file or select a saved dataset from the tabs above.")
else:
    st.warning("⚠️ Please create or select a project above to begin uploading and analyzing data.")

st.markdown('<div class="footer">© 2026 Zeiss LSM980 FCS Analysis Suite | Developed by SHIBASISH | Confidential & Proprietary</div>', unsafe_allow_html=True)