import streamlit as st
import pandas as pd
import json
import plotly.express as px
from openai import OpenAI

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & LEGAL TERMS OF SERVICE GATE
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="TunerAI - ECU Diagnostic & Learning Assistant",
    page_icon="🏎️",
    layout="wide"
)

# Initialize Session State for Terms Agreement & Subscription
if "tos_agreed" not in st.session_state:
    st.session_state.tos_agreed = False

if "is_subscribed" not in st.session_state:
    st.session_state.is_subscribed = False

st.title("🏎️ TunerAI: ECU Datalog Safety & Diagnostic Assistant")
st.caption("AI-Powered Engine Calibration Education & Deterministic Safety Analysis")

# ------------------------------------------------------------------------------
# MANDATORY LEGAL TERMS OF SERVICE MODAL / DIALOG
# ------------------------------------------------------------------------------
if not st.session_state.tos_agreed:
    st.error("⚠️ TERMS OF SERVICE & LIABILITY RELEASE REQUIRED")
    
    st.markdown("""
    ### Terms of Service & User Release of Liability
    
    Please read and accept the following terms before using TunerAI:
    
    1. **EDUCATIONAL & OFF-ROAD USE ONLY:** TunerAI is designed solely for educational, competition, and off-road analysis. It is not intended for street-driven emissions-controlled vehicles.
    2. **NO FLASH BINARY CREATION:** TunerAI does NOT generate executable calibration files (`.bin`, `.hex`, `.cal`). It provides diagnostic analysis only.
    3. **USER ASSUMPTION OF RISK:** Automotive engine calibration carries inherent mechanical risks, including severe engine damage or fire. You assume full responsibility for inspecting physical hardware and manually verifying all engine parameters.
    4. **LIMITATION OF LIABILITY:** The creators and operators of TunerAI shall not be held liable for any direct, indirect, or consequential engine or component damage resulting from telemetry interpretation.
    """)
    
    if st.button("I Agree to the Terms of Service & Release of Liability"):
        st.session_state.tos_agreed = True
        st.rerun()
    else:
        st.stop()

st.success("✅ Terms of Service Accepted")

# ------------------------------------------------------------------------------
# 2. MULTI-ECU LOG COLUMN AUTO-MAPPER
# ------------------------------------------------------------------------------
class ECULogNormalizer:
    """Normalizes column headers from various ECU software into unified names."""
    
    COLUMN_MAPS = {
        'RPM': ['rpm', 'engine_speed', 'engine speed', 'RPM'],
        'MAP_kPa': ['map', 'map_kpa', 'manifold_pressure', 'boost_pressure', 'MAP_kPa', 'Boost'],
        'AFR': ['afr', 'afr_actual', 'wideband_afr', 'afr1', 'AFR', 'Air Fuel Ratio'],
        'Injector_Duty': ['idc', 'injector_duty', 'duty_cycle_1', 'injector_duty_cycle', 'Injector_Duty'],
        'Knock_Retard': ['knock', 'knock_retard', 'feedback_knock', 'total_knock_retard', 'Knock_Retard'],
        'IAT_C': ['iat', 'iat_c', 'intake_air_temp', 'air_temp', 'IAT_C'],
        'Oil_Press_PSI': ['oil_press', 'oil_pressure', 'oil_press_psi', 'Oil_Press_PSI']
    }

    @classmethod
    def normalize(cls, df: pd.DataFrame) -> pd.DataFrame:
        normalized_df = df.copy()
        renamed_cols = {}
        
        for standard_col, aliases in cls.COLUMN_MAPS.items():
            for col in df.columns:
                if col.strip().lower() in [a.lower() for a in aliases]:
                    renamed_cols[col] = standard_col
                    break
        
        return normalized_df.rename(columns=renamed_cols)

# ------------------------------------------------------------------------------
# 3. DETERMINISTIC HARDWARE FAILSAFE ENGINE
# ------------------------------------------------------------------------------
class EngineSafetyGuardrail:
    """Scans normalized datalogs for physical safety thresholds before AI analysis."""
    
    def evaluate_log(self, df: pd.DataFrame, is_turbo: bool) -> dict:
        flags = []
        status = "PASSED"

        if 'Injector_Duty' in df.columns:
            max_idc = df['Injector_Duty'].max()
            if max_idc > 90.0:
                status = "CRITICAL"
                flags.append(f"🔴 CRITICAL: Injector Duty Cycle reached {max_idc:.1f}%. Risk of static fuel starvation!")
            elif max_idc > 85.0:
                if status != "CRITICAL": status = "WARNING"
                flags.append(f"🟡 WARNING: Injector Duty Cycle reached {max_idc:.1f}%. Approaching headroom limit.")

        if is_turbo and 'MAP_kPa' in df.columns and 'AFR' in df.columns:
            boosted = df[df['MAP_kPa'] > 115]
            if not boosted.empty:
                max_boost_afr = boosted['AFR'].max()
                if max_boost_afr > 12.2:
                    status = "CRITICAL"
                    flags.append(f"🔴 CRITICAL: Lean AFR of {max_boost_afr:.1f}:1 detected under boost (>115 kPa)!")

        if 'Knock_Retard' in df.columns:
            max_knock = df['Knock_Retard'].max()
            if max_knock > 3.0:
                status = "CRITICAL"
                flags.append(f"🔴 CRITICAL: Knock sensor pulled {max_knock:.1f}° of timing due to severe detonation.")
            elif max_knock > 1.0:
                if status != "CRITICAL": status = "WARNING"
                flags.append(f"🟡 WARNING: Minor knock detected ({max_knock:.1f}° retard).")

        if 'IAT_C' in df.columns:
            max_iat = df['IAT_C'].max()
            if max_iat > 55.0:
                status = "CRITICAL"
                flags.append(f"🔴 CRITICAL: Intake Air Temp reached {max_iat:.1f}°C (131°F). Extreme heat soak / detonation risk!")
            elif max_iat > 45.0:
                if status != "CRITICAL": status = "WARNING"
                flags.append(f"🟡 WARNING: Elevated Intake Air Temp ({max_iat:.1f}°C / 113°F). Consider pulling timing.")

        if 'Oil_Press_PSI' in df.columns and 'RPM' in df.columns:
            high_rpm = df[df['RPM'] > 4000]
            if not high_rpm.empty:
                min_oil_press = high_rpm['Oil_Press_PSI'].min()
                if min_oil_press < 30.0:
                    status = "CRITICAL"
                    flags.append(f"🔴 CRITICAL: Low oil pressure ({min_oil_press:.1f} PSI) above 4000 RPM! Bearing failure risk!")

        return {"status": status, "flags": flags}

# ------------------------------------------------------------------------------
# 4. SIDEBAR: SUBSCRIPTION & VEHICLE PROFILE
# ------------------------------------------------------------------------------
st.sidebar.header("💳 Membership & Subscription")

# Custom Stripe Link Placeholder
STRIPE_PAYMENT_LINK = "https://buy.stripe.com/your_custom_stripe_link_here"

if not st.session_state.is_subscribed:
    st.sidebar.warning("🔒 Free Tier: Basic Safety Check Only")
    st.sidebar.markdown(f"[👉 Upgrade to TunerAI Pro ($19/mo)]({STRIPE_PAYMENT_LINK})", unsafe_allow_html=True)
    
    # Dev Toggle to Simulate Active Subscription
    if st.sidebar.checkbox("Simulate Paid Subscription (Dev Mode)"):
        st.session_state.is_subscribed = True
        st.rerun()
else:
    st.sidebar.success("⭐ TunerAI Pro Subscriber")

st.sidebar.divider()
st.sidebar.header("1. Vehicle & Setup Profile")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key to enable AI analysis.")

make = st.sidebar.text_input("Make/Model", "Mazda Miata 1.8L")
aspiration = st.sidebar.selectbox("Aspiration", ["Turbocharged / Supercharged", "Naturally Aspirated"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["91 AKI (95 RON)", "93 AKI (98 RON)", "E85 / Flex Fuel"])
is_turbo = aspiration == "Turbocharged / Supercharged"

# ------------------------------------------------------------------------------
# 5. MAIN INTERFACE: DATALOG INPUT & EVALUATION
# ------------------------------------------------------------------------------
st.header("2. Datalog Input")
use_demo = st.checkbox("⚡ Use Demo Datalog (Instant Test)", value=False)

df_raw = None

if use_demo:
    sample_data = {
        'Engine Speed': [2000, 3000, 4000, 5000, 6000, 6500],
        'Boost Pressure': [40, 80, 120, 145, 155, 160],
        'Wideband_AFR': [14.7, 13.5, 12.8, 12.9, 12.4, 11.8],
        'Duty_Cycle_1': [25.0, 45.0, 68.0, 89.0, 92.5, 94.0],
        'Total_Knock_Retard': [0.0, 0.0, 0.5, 3.2, 1.5, 0.0],
        'Intake_Air_Temp': [32.0, 35.0, 42.0, 48.0, 52.0, 56.0],
        'Oil_Pressure_PSI': [45.0, 50.0, 42.0, 35.0, 28.0, 25.0]
    }
    df_raw = pd.DataFrame(sample_data)
    st.success("Loaded Unmapped Demo Datalog (Simulating Megasquirt/HP Tuners format)!")
else:
    uploaded_file = st.file_uploader("Upload CSV Datalog (Megasquirt, HP Tuners, Cobb, etc.)", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)

if df_raw is not None:
    # Run Auto-Mapper
    normalizer = ECULogNormalizer()
    df = normalizer.normalize(df_raw)

    tab1, tab2, tab3 = st.tabs(["🛡️ Safety Evaluation", "📊 Interactive Visualizations", "📋 Beta Audit Checklist"])

    with tab1:
        st.subheader("Raw Telemetry Preview (Mapped Columns)")
        st.dataframe(df.head())

        guardrail = EngineSafetyGuardrail()
        safety_results = guardrail.evaluate_log(df, is_turbo=is_turbo)

        st.subheader("Deterministic Failsafe Analysis")
        if safety_results["status"] == "CRITICAL":
            st.error("🔴 CRITICAL HARDWARE SAFETY BREACH DETECTED")
        elif safety_results["status"] == "WARNING":
            st.warning("🟡 SAFETY WARNINGS DETECTED")
        else:
            st.success("🟢 ALL DETERMINISTIC GUARDRAILS PASSED")

        for flag in safety_results["flags"]:
            st.write(flag)

        st.subheader("AI Educational Diagnostic & Lessons")
        if not st.session_state.is_subscribed:
            st.info("🔒 Premium AI Diagnostic Lessons require a TunerAI Pro Subscription. Upgrade in the sidebar to unlock.")
        else:
            if not api_key:
                st.warning("Please enter your OpenAI API key in the sidebar to run the AI Diagnostic Tutor.")
            else:
                if st.button("Run AI Diagnostic Analysis"):
                    with st.spinner("Analyzing log patterns and calculating thermodynamic responses..."):
                        client = OpenAI(api_key=api_key)
                        
                        system_prompt = f"""
                        You are the Tuning Educator AI. Analyze the provided ECU datalog summary for a {make} running on {fuel_type}.
                        Aspiration: {aspiration}.
                        Deterministic Flags Reported: {json.dumps(safety_results['flags'])}
                        
                        STRICT COMPLIANCE RULES:
                        1. Focus purely on explaining the 'WHY' behind the data (thermodynamics, air/fuel ratios, ignition timing).
                        2. DO NOT provide step-by-step remapping instructions or binary file outputs.
                        3. ALWAYS remind the user to check physical hardware (fuel filters, spark plugs, boost leaks).
                        """
                        
                        user_prompt = f"Datalog Summary Stats:\n{df.describe().to_string()}"
                        
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.3
                        )
                        
                        st.markdown("### AI Diagnostic Insights")
                        st.write(response.choices[0].message.content)

    with tab2:
        st.subheader("Interactive Telemetry Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'RPM' in df.columns and 'AFR' in df.columns:
                fig_afr = px.line(df, y='AFR', x='RPM', title="AFR Curve vs Engine RPM", markers=True)
                fig_afr.add_hline(y=12.2, line_dash="dash", line_color="red", annotation_text="Lean Boost Threshold")
                st.plotly_chart(fig_afr, use_container_width=True)

        with col2:
            if 'RPM' in df.columns and 'Injector_Duty' in df.columns:
                fig_idc = px.bar(df, x='RPM', y='Injector_Duty', title="Injector Duty Cycle (%) vs RPM")
                fig_idc.add_hline(y=85.0, line_dash="dash", line_color="orange", annotation_text="Warning (85%)")
                fig_idc.add_hline(y=90.0, line_dash="dash", line_color="red", annotation_text="Critical (90%)")
                st.plotly_chart(fig_idc, use_container_width=True)

        if 'Oil_Press_PSI' in df.columns and 'RPM' in df.columns:
            fig_oil = px.line(df, x='RPM', y='Oil_Press_PSI', title="Oil Pressure (PSI) vs RPM", markers=True)
            fig_oil.add_hline(y=30.0, line_dash="dash", line_color="red", annotation_text="Min Pressure Floor (30 PSI)")
            st.plotly_chart(fig_oil, use_container_width=True)

    with tab3:
        st.subheader("Beta Tester & Engineering Verification Checklist")
        st.markdown("""
        **For Review by Imaad Shaik & Engineering Beta Testers:**
        
        - [x] **ToS Modal Gate:** Terms of service agreement required before viewing interface.
        - [x] **Multi-ECU Header Mapper:** Automatically standardizes Megasquirt, HP Tuners, and Cobb column names.
        - [x] **Subscription Paywall:** Gates AI diagnostic analysis behind Pro membership.
        - [ ] **Physical Hardware Verification:** Confirm safety rules with Imaad on live dyno logs.
        """)

# ------------------------------------------------------------------------------
# 6. FOOTER
# ------------------------------------------------------------------------------
st.divider()
st.caption("TunerAI Commercial Edition v0.3 | Off-Road & Competition Use Only")
