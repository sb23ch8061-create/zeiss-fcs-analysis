import streamlit as st
import numpy as np
import pandas as pd
import io
import scipy.constants as const
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from read_zeiss import load_zeiss
from fit_models import fit_standard, fit_triplet, fit_anomalous, fit_two_component
from export_results import results_to_dataframe
from auth_db import (
    verify_user, 
    create_user, 
    create_project, 
    get_user_projects, 
    save_dataset_record, 
    get_project_datasets
)

# Wrapper to seamlessly handle all 4 FCS models
def fit_fcs_model(tau, G, model_type="Standard 3D", S=7.5):
    tau = np.asarray(tau)
    G = np.asarray(G)
    n = len(G)

    if model_type == "Triplet 3D":
        res = fit_triplet(tau, G, S=S)
        nparam = 4
    elif model_type == "Anomalous 3D":
        res = fit_anomalous(tau, G, S=S)
        nparam = 3
    elif model_type == "2-Component 3D":
        res = fit_two_component(tau, G, S=S)
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

# Page Setup
st.set_page_config(page_title="Zeiss LSM980 FCS Suite Pro", layout="wide", page_icon="🔬")

# UI Custom Styling
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
    .stMetric {
        background-color: #1A1D24;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #262730;
    }
    </style>
""", unsafe_allow_html=True)

# State Management
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "current_project" not in st.session_state:
    st.session_state.current_project = None
if "omega_sq" not in st.session_state:
    st.session_state.omega_sq = 0.042

# --- USER AUTHENTICATION SCREEN ---
def login_screen():
    st.title("🔬 Zeiss LSM980 FCS Enterprise Platform")
    st.subheader("Multi-Model FCS Fitting & Publication-Grade Workspace")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Register Account"])
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username / Lab ID")
                password = st.text_input("Password", type="password")
                submit_btn = st.form_submit_button("Enter Suite", type="primary")
                if submit_btn:
                    if verify_user(username.strip(), password):
                        st.session_state.authenticated = True
                        st.session_state.username = username.strip()
                        st.success("Authentication successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
                        
        with tab_register:
            with st.form("register_form"):
                new_username = st.text_input("Choose Username / Lab ID")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                reg_btn = st.form_submit_button("Create Account")
                if reg_btn:
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        success, msg = create_user(new_username.strip(), new_password)
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                
    with col2:
        st.info("""
        **Platform Capabilities:**
        * **4 Models:** Standard 3D, Triplet 3D, Anomalous 3D, 2-Component 3D.
        * **Interactive Plotly Suite:** Scroll-to-zoom & sub-millisecond residual inspection.
        * **AIC/BIC Criteria:** Automated statistical model evaluation.
        * **Export Options:** Publication reports, interactive HTML plots, and CSV tables.
        """)

if not st.session_state.authenticated:
    login_screen()
    st.markdown('<div class="footer">© 2026 Zeiss LSM980 FCS Suite Pro | Proprietary & Confidential</div>', unsafe_allow_html=True)
    st.stop()

# =====================================================================
# --- MAIN SUITE INTERFACE ---
# =====================================================================
st.sidebar.markdown(f"👤 **Active User:** `{st.session_state.username}`")
if st.sidebar.button("Log Out"):
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.current_project = None
    st.rerun()

st.title("Zeiss LSM980 FCS Real-Time Dashboard")

# --- PROJECT WORKSPACE ---
st.subheader("📁 Project Workspace")
project_col1, project_col2 = st.columns([1, 2])

user_projects = get_user_projects(st.session_state.username)
project_names = [p["name"] for p in user_projects]

with project_col1:
    with st.form("new_project_form", clear_on_submit=True):
        new_proj_name = st.text_input("Create New Project", placeholder="e.g., Membrane Dynamics")
        create_proj_btn = st.form_submit_button("Create")
        if create_proj_btn and new_proj_name:
            success, msg = create_project(st.session_state.username, new_proj_name)
            if success:
                st.session_state.current_project = new_proj_name
                st.rerun()

with project_col2:
    if project_names:
        default_idx = project_names.index(st.session_state.current_project) if st.session_state.current_project in project_names else 0
        selected_project = st.selectbox("Active Workspace", ["-- Select a Project --"] + project_names, index=(default_idx + 1) if st.session_state.current_project else 0)
        if selected_project != "-- Select a Project --":
            if st.session_state.current_project != selected_project:
                st.session_state.current_project = selected_project
                st.rerun()
        else:
            st.session_state.current_project = None

st.divider()

if st.session_state.current_project:
    st.markdown(f"### Current Workspace: **{st.session_state.current_project}**")
    project_dir = Path("user_data") / st.session_state.username / st.session_state.current_project
    project_dir.mkdir(parents=True, exist_ok=True)
    
    saved_datasets = get_project_datasets(st.session_state.username, st.session_state.current_project)
    
    source_tab1, source_tab2, source_tab3 = st.tabs(["📤 Upload Raw File", "📂 Project Datasets", "📊 Saved Analysis Reports"])
    
    active_file_stream = None
    active_filename = None
    
    with source_tab1:
        uploaded_file = st.file_uploader("📂 Upload Zeiss Raw FCS Data (.txt, .dat, .csv)", type=["txt", "dat", "csv"])
        if uploaded_file is not None:
            saved_path = project_dir / uploaded_file.name
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            save_dataset_record(st.session_state.username, st.session_state.current_project, uploaded_file.name, str(saved_path))
            active_file_stream = saved_path
            active_filename = uploaded_file.name
            st.success(f"Dataset **{uploaded_file.name}** uploaded successfully.")

    with source_tab2:
        if saved_datasets:
            dataset_options = {d["filename"]: Path(d["file_path"]) for d in saved_datasets}
            selected_saved = st.selectbox("Select saved dataset:", list(dataset_options.keys()))
            if selected_saved:
                active_file_stream = dataset_options[selected_saved]
                active_filename = selected_saved
        else:
            st.info("No saved datasets available.")

    with source_tab3:
        csv_files = list(project_dir.glob("*_results.csv"))
        if csv_files:
            st.markdown("#### 🗄️ Saved Analysis Tables & Reports")
            for f in csv_files:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.write(f"📄 **{f.name}**")
                with c2:
                    with open(f, "rb") as file:
                        st.download_button("Download CSV", data=file, file_name=f.name, mime="text/csv", key=f.name)
        else:
            st.info("No saved reports found.")

    @st.cache_data
    def get_data(file_path):
        try:
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
            st.sidebar.header("⚙️ Model Configuration")
            s_value = st.sidebar.slider("Structure Parameter (S)", min_value=1.0, max_value=15.0, value=7.5, step=0.1)
            
            # Upgraded Model Selection (All 4 Models)
            model_options = ["Standard 3D", "Triplet 3D", "Anomalous 3D", "2-Component 3D"]
            model_choice = st.sidebar.selectbox("Select FCS Model", model_options)
            
            selected_reps = st.sidebar.multiselect("Compare Repetitions", options=rep_keys, default=rep_keys[:2] if len(rep_keys) >= 2 else rep_keys)

            st.sidebar.divider()
            st.sidebar.subheader("🛡️ Data Filters")
            min_r2 = st.sidebar.slider("Minimum R² Threshold Filter", 0.80, 0.999, 0.90, 0.005)

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

            # --- FIT ENGINE ---
            results = {}
            with st.spinner(f"Fitting data using [{model_choice}]..."):
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

            # --- INTERACTIVE PLOTLY GRAPHS ---
            st.markdown("*(Tip: Use your scroll wheel to zoom in on any section of the curves)*")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Average Curve Fit & Residuals")
                fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)
                
                fig1.add_trace(go.Scatter(x=avg_tau, y=avg_G, mode='markers', name='Experiment (Avg)', marker=dict(color='gray', opacity=0.4, size=5)), row=1, col=1)
                fig1.add_trace(go.Scatter(x=avg_tau[avg_mask], y=avg_res["fit"], mode='lines', name=f'Fit ({model_choice})', line=dict(color='red', width=2.5)), row=1, col=1)
                
                residuals = avg_G[avg_mask] - avg_res["fit"]
                fig1.add_trace(go.Scatter(x=avg_tau[avg_mask], y=residuals, mode='lines', name='Residuals', line=dict(color='#0055FF', width=1.5)), row=2, col=1)
                fig1.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
                
                fig1.update_xaxes(type="log", row=1, col=1)
                fig1.update_xaxes(type="log", title_text="Lag time (s)", row=2, col=1)
                fig1.update_yaxes(title_text="G(tau)", row=1, col=1)
                fig1.update_yaxes(title_text="Residuals", row=2, col=1)
                fig1.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), showlegend=True, hovermode="x unified")
                
                st.plotly_chart(fig1, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False})
                
            with col2:
                st.subheader("Selected Repetition Fits")
                if selected_reps:
                    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)
                    colors = px.colors.qualitative.Plotly
                    
                    for i, rep_name in enumerate(selected_reps):
                        if rep_name in results:
                            rep_tau = np.array(datasets[rep_name]["tau"])
                            rep_G = np.array(datasets[rep_name]["G"])
                            rep_mask = rep_tau >= 1e-6
                            rep_res = results[rep_name]
                            color = colors[i % len(colors)]
                            
                            fig2.add_trace(go.Scatter(x=rep_tau, y=rep_G, mode='markers', name=f'{rep_name}', marker=dict(color=color, opacity=0.3, size=4), showlegend=False), row=1, col=1)
                            fig2.add_trace(go.Scatter(x=rep_tau[rep_mask], y=rep_res["fit"], mode='lines', name=rep_name, line=dict(color=color, width=2)), row=1, col=1)
                            fig2.add_trace(go.Scatter(x=rep_tau[rep_mask], y=rep_G[rep_mask] - rep_res["fit"], mode='lines', name=f'{rep_name} Res', line=dict(color=color, width=1.5), showlegend=False), row=2, col=1)
                    
                    fig2.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
                    fig2.update_xaxes(type="log", row=1, col=1)
                    fig2.update_xaxes(type="log", title_text="Lag time (s)", row=2, col=1)
                    fig2.update_yaxes(title_text="G(tau)", row=1, col=1)
                    fig2.update_yaxes(title_text="Residuals", row=2, col=1)
                    fig2.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), showlegend=True, hovermode="x unified")
                    
                    st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False})

            st.divider()

            # --- PHYSICAL PARAMETERS ---
            st.subheader("Summary Statistics & Biophysical Parameters")
            
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

                st.dataframe(df_results, height=300, use_container_width=True)

            st.subheader("Model Goodness-of-Fit Criteria (Average Dataset)")
            aic_col, bic_col, chi_col, r2_col = st.columns(4)
            aic_col.metric("Akaike Info Criterion (AIC)", f"{avg_res['AIC']:.2f}")
            bic_col.metric("Bayesian Info Criterion (BIC)", f"{avg_res['BIC']:.2f}")
            chi_col.metric("Reduced Chi²", f"{avg_res['Chi2']:.6f}")
            r2_col.metric("Coefficient of Determination (R²)", f"{avg_res['R2']:.5f}")

            st.divider()

            # --- PUBLICATION REPORT & PERSISTENCE ---
            st.subheader("💾 Save Session & Generate Publication Report")
            save_prefix = st.text_input("Session / File Prefix:", value=f"{active_filename}_fitted" if active_filename else "FCS_Analysis")

            dl1, dl2, dl3 = st.columns([1, 1, 1])
            with dl1:
                if 'df_results' in locals():
                    st.download_button("📊 Download Full CSV Data", data=df_results.to_csv(index=False).encode('utf-8'), file_name=f"{save_prefix}_stats.csv", mime='text/csv')
            
            with dl2:
                # Publication HTML Report Generator
                report_html = f"""
                <html>
                <head>
                    <title>Zeiss FCS Analysis Report - {save_prefix}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 30px; }}
                        h1 {{ color: #003366; }}
                        table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    <h1>Zeiss LSM980 FCS Analysis Report</h1>
                    <p><strong>Dataset:</strong> {active_filename}</p>
                    <p><strong>Model Applied:</strong> {model_choice}</p>
                    <p><strong>User:</strong> {st.session_state.username}</p>
                    <h2>Key Biophysical Summary</h2>
                    <ul>
                        <li><strong>Mean Diffusion Time (&tau;<sub>D</sub>):</strong> {np.mean(tauD_us_list):.2f} &mu;s</li>
                        <li><strong>Mean Diffusivity (D):</strong> {np.mean(D_list_um2s):.2f} &mu;m&sup2;/s</li>
                        <li><strong>Mean Hydrodynamic Radius (R<sub>h</sub>):</strong> {np.mean(Rh_nm_list):.3f} nm</li>
                        <li><strong>Mean Concentration:</strong> {np.mean(Conc_nM_list):.3f} nM</li>
                    </ul>
                    <h2>Model Quality</h2>
                    <p><strong>R&sup2;:</strong> {avg_res['R2']:.5f} | <strong>Chi&sup2;:</strong> {avg_res['Chi2']:.6f} | <strong>AIC:</strong> {avg_res['AIC']:.2f} | <strong>BIC:</strong> {avg_res['BIC']:.2f}</p>
                </body>
                </html>
                """
                st.download_button("📄 Download Publication Report (.html)", data=report_html, file_name=f"{save_prefix}_Report.html", mime="text/html")

            with dl3:
                if st.button("💾 Save Session to Account"):
                    if 'df_results' in locals():
                        save_path = project_dir / f"{save_prefix}_results.csv"
                        df_results.to_csv(save_path, index=False)
                        st.success(f"Saved **{save_prefix}_results.csv** to project folder!")

    else:
        st.info("👆 Upload or select a dataset to begin.")
else:
    st.warning("⚠️ Select or create a project above.")

st.markdown('<div class="footer">© 2026 Zeiss LSM980 FCS Suite Pro | Confidential & Proprietary</div>', unsafe_allow_html=True)