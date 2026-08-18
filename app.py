import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import scipy.constants as const
from read_zeiss import load_zeiss
from fit_models import fit_fcs_model
from export_results import results_to_dataframe

# Page Configuration
st.set_page_config(page_title="Zeiss LSM980 FCS Suite", layout="wide", page_icon="🔬")

# Custom CSS for Copyright Watermark Footer & Dark Theme Styling
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
if "user_data_store" not in st.session_state:
    st.session_state.user_data_store = {}
if "omega_sq" not in st.session_state:
    st.session_state.omega_sq = 0.042

# --- USER AUTHENTICATION MODULE ---
def login_screen():
    st.title("🔬 Zeiss LSM980 FCS Real-Time Analysis Suite")
    st.subheader("User Authentication & Data Workspace")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Sign In / Register Session")
        username = st.text_input("Username or Lab ID")
        password = st.text_input("Access Key / Password", type="password")
        
        if st.button("Enter Analysis Suite", type="primary"):
            if username.strip() != "":
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Please provide a valid Username or Lab ID.")
                
    with col2:
        st.info("""
        **Platform Features:**
        * Multi-Model FCS Fitting (1-Comp, 2-Comp, Anomalous, Triplet, 2D).
        * Automated Model Selection (AIC & BIC criteria).
        * Stokes-Einstein Hydrodynamic Radius ($R_h$) & Molar Concentration ($C$) Calculators.
        * File upload & personal analysis session storage.
        """)

if not st.session_state.authenticated:
    login_screen()
    # Watermark Footer
    st.markdown('<div class="footer">© 2026 Zeiss LSM980 FCS Analysis Suite | Developed by Ram et. al. | Confidential & Proprietary</div>', unsafe_allow_html=True)
    st.stop()

# --- MAIN APPLICATION WORKSPACE ---
st.sidebar.markdown(f"👤 **User:** `{st.session_state.username}`")
if st.sidebar.button("Log Out"):
    st.session_state.authenticated = False
    st.rerun()

st.title("Zeiss LSM980 FCS Real-Time Dashboard")

# 1. File Upload & Workspace
uploaded_file = st.file_uploader(
    "📂 Upload Zeiss FCS Raw Data File (.txt, .dat, .csv)", 
    type=["txt", "dat", "csv"]
)

@st.cache_data
def get_data(file_obj):
    try:
        return load_zeiss(file_obj)
    except Exception as e:
        st.error(f"Error reading file stream: {e}")
        return None

if uploaded_file is not None:
    datasets = get_data(uploaded_file)

    if datasets:
        rep_keys = [k for k in datasets.keys() if k != "Average"]

        # --- SIDEBAR CONTROLS ---
        st.sidebar.header("⚙️ Model & Fit Parameters")
        
        s_value = st.sidebar.slider("Structure Parameter (S)", min_value=1.0, max_value=15.0, value=7.5, step=0.1)
        
        model_options = [
            "Standard 3D", 
            "Triplet 3D", 
            "2-Component 3D", 
            "2-Component + Triplet 3D", 
            "Anomalous 3D", 
            "2D Membrane", 
            "2D + Triplet"
        ]
        model_choice = st.sidebar.selectbox("Select FCS Model", model_options)
        
        selected_reps = st.sidebar.multiselect(
            "Select Repetitions to Compare", 
            options=rep_keys, 
            default=rep_keys[:2] if len(rep_keys) >= 2 else rep_keys
        )

        # Outlier Filter Options
        st.sidebar.divider()
        st.sidebar.subheader("🛡️ Data Quality Controls")
        min_r2 = st.sidebar.slider("Minimum R² Threshold Filter", 0.80, 0.999, 0.95, 0.005)

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
            viscosity_mPas = st.number_input("Solvent Viscosity η (mPa·s)", value=0.890) # Water at 25C
            
            temp_k = temp_c + 273.15
            viscosity_pas = viscosity_mPas * 1e-3

        # --- 2. FIT PROCESSING ENGINE ---
        results = {}
        valid_reps = []

        with st.spinner(f"Fitting data using model [{model_choice}]..."):
            for name in rep_keys:
                tau = np.array(datasets[name]["tau"])
                G = np.array(datasets[name]["G"])
                
                mask = tau >= 1e-6
                res = fit_fcs_model(tau[mask], G[mask], model_type=model_choice, S=s_value)
                
                if res["R2"] >= min_r2:
                    results[name] = res
                    valid_reps.append(name)

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
        
        tauD_sec_list = np.array([r["tauD"] for r in results.values()])
        tauD_us_list = tauD_sec_list * 1e6
        D_list_um2s = st.session_state.omega_sq / (4.0 * tauD_sec_list)
        D_list_m2s = D_list_um2s * 1e-12

        # Stokes-Einstein Hydrodynamic Radius (Rh = kBT / (6*pi*eta*D))
        Rh_nm_list = (const.k * temp_k) / (6.0 * np.pi * viscosity_pas * D_list_m2s) * 1e9

        # Effective Volume Veff = pi^(3/2) * S * w0^3 where w0^2 = omega_sq
        w0_um = np.sqrt(st.session_state.omega_sq)
        V_eff_L = (np.pi ** 1.5) * s_value * (w0_um ** 3) * 1e-15 # Convert um^3 to Liters
        
        N_list = np.array([r["N"] for r in results.values()])
        Conc_nM_list = (N_list / (V_eff_L * const.Avogadro)) * 1e9 # nM

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Diffusion Time (τD)", f"{np.mean(tauD_us_list):.2f} ± {np.std(tauD_us_list):.2f} µs")
        m2.metric("Diffusivity (D)", f"{np.mean(D_list_um2s):.2f} ± {np.std(D_list_um2s):.2f} µm²/s")
        m3.metric("Hydrodynamic Radius (Rh)", f"{np.mean(Rh_nm_list):.3f} ± {np.std(Rh_nm_list):.3f} nm")
        m4.metric("Molar Concentration (C)", f"{np.mean(Conc_nM_list):.3f} ± {np.std(Conc_nM_list):.3f} nM")

        # Build Extended Output DataFrame
        df_results = results_to_dataframe(list(results.values()))
        df_results["Diffusivity D (µm²/s)"] = D_list_um2s
        df_results["Rh (nm)"] = Rh_nm_list
        df_results["Concentration (nM)"] = Conc_nM_list

        st.dataframe(df_results, height=350, use_container_width=True)

        # Model Selection Evaluation Metrics (AIC & BIC)
        st.subheader("Model Selection & Information Criteria (Average Dataset)")
        aic_col, bic_col, chi_col, r2_col = st.columns(4)
        aic_col.metric("Akaike Info Criterion (AIC)", f"{avg_res['AIC']:.2f}")
        bic_col.metric("Bayesian Info Criterion (BIC)", f"{avg_res['BIC']:.2f}")
        chi_col.metric("Reduced Chi²", f"{avg_res['Chi2']:.6f}")
        r2_col.metric("Coefficient of Determination (R²)", f"{avg_res['R2']:.5f}")

        st.divider()

        # --- 5. DATA PERSISTENCE & EXPORT SECTION ---
        st.subheader("💾 Save, Store & Export Session Analysis")
        save_prefix = st.text_input("Filename / Session Name Prefix:", value="FCS_Analysis")

        dl1, dl2, dl3, dl4 = st.columns(4)
        with dl1:
            st.download_button("📊 Download Full CSV Data", data=df_results.to_csv(index=False).encode('utf-8'), file_name=f"{save_prefix}_stats.csv", mime='text/csv')
        with dl2:
            st.download_button("📈 Download Average Graph", data=buf1, file_name=f"{save_prefix}_AvgFit.png", mime='image/png')
        with dl3:
            if buf2:
                st.download_button("📉 Download Repetition Graph", data=buf2, file_name=f"{save_prefix}_RepsFit.png", mime='image/png')
        with dl4:
            if st.button("💾 Save Session to Account"):
                st.session_state.user_data_store[save_prefix] = df_results
                st.success(f"Saved session '{save_prefix}' to user account '{st.session_state.username}'.")

else:
    st.info("👆 Please upload a raw FCS correlation text file to unlock real-time fitting.")

# --- PERSISTENT FOOTER ---
st.markdown('<div class="footer">© 2026 Zeiss LSM980 FCS Analysis Suite | Developed by Ram et. al. | Confidential & Proprietary</div>', unsafe_allow_html=True)