import streamlit as st
import pandas as pd
import json
import plotly.express as px
from openai import OpenAI

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & LEGAL DISCLAIMERS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="TunerAI - ECU Diagnostic & Learning Assistant",
    page_icon="🏎️",
    layout="wide"
)

EDUCATIONAL_DISCLAIMER = (
    "**NOTICE & LEGAL DISCLAIMER:** This software is for **EDUCATIONAL AND COMPETITION "
    "/ OFF-ROAD USE ONLY**. Engine tuning carries inherent risks of severe mechanical damage. "
    "This tool does not generate executable flash files. Always inspect engine hardware "
    "and verify all calibration changes manually before modifying any vehicle parameters."
)

st.title("🏎️ TunerAI: ECU Datalog Safety & Diagnostic Assistant")
st.caption("AI-Powered Engine Calibration Education & Deterministic Safety Analysis")
st.info(EDUCATIONAL_DISCLAIMER)

# ------------------------------------------------------------------------------
# 2. ENHANCED DETERMINISTIC HARDWARE FAILSAFE ENGINE
# ------------------------------------------------------------------------------
class EngineSafetyGuardrail:
    """Scans datalogs for physical safety thresholds before AI analysis."""
    
    def evaluate_log(self, df: pd.DataFrame, is_turbo: bool) -> dict:
        flags = []
        status = "PASSED"

        # Rule 1: Injector Duty Cycle Check
        if 'Injector_Duty' in df.columns:
            max_idc = df['Injector_Duty'].max()
            if max_idc > 90.0:
                status = "CRITICAL"
                flags.append(f"🔴 CRITICAL: Injector Duty Cycle reached {max_idc:.1f}%. Risk of static fuel starvation!")
            elif max_idc > 85.0:
                if status != "CRITICAL": status = "WARNING"
                flags.append(f"🟡 WARNING: Injector Duty Cycle reached {max_idc:.1f}%. Approaching headroom limit.")

        # Rule 2: Lean under boost check (MAP > 115 kPa = Boost)
        if is_turbo and 'MAP_kPa' in df.columns and 'AFR' in df.columns:
            boosted = df[df['MAP_kPa'] > 115]
            if not boosted.empty:
                max_boost_afr = boosted['AFR'].max()
                if max_boost_afr > 12.2:
                    status = "CRITICAL"
                    flags.append(f"🔴 CRITICAL: Lean AFR of {max_boost_afr:.1f}:1 detected under boost (>115 kPa)!")

        # Rule 3: Knock Retard Check
        if 'Knock_Retard' in df.columns:
            max_knock = df['Knock_Retard'].max()
            if max_knock > 3.0:
                status = "CRITICAL"
                flags.append(f"🔴 CRITICAL: Knock sensor pulled {max_knock:.1f}° of timing due to severe detonation.")
            elif max_knock > 1.0:
                if status != "CRITICAL": status = "WARNING"
                flags.append(f"🟡 WARNING: Minor knock detected ({max_knock:.1f}° retard).")

        # Rule 4: Intake Air Temp (IAT) Thermal Limit
        if 'IAT_C' in df.columns:
            max_iat = df['IAT_C'].max()
            if max_iat > 55.0:
                status = "CRITICAL"
                flags.append(f"🔴 CRITICAL: Intake Air Temp reached {max_iat:.1f}°C (131°F). Extreme heat soak / detonation risk!")
            elif max_iat > 45.0:
                if status != "CRITICAL": status = "WARNING"
                flags.append(f"🟡 WARNING: Elevated Intake Air Temp ({max_iat:.1f}°C / 113°F). Consider pulling timing.")

        # Rule 5: Low Oil Pressure Under Load Check
        if 'Oil_Press_PSI' in df.columns and 'RPM' in df.columns:
            high_rpm = df[df['RPM'] > 4000]
            if not high_rpm.empty:
                min_oil_press = high_rpm['Oil_Press_PSI'].min()
                if min_oil_press < 30.0:
                    status = "CRITICAL"
                    flags.append(f"🔴 CRITICAL: Low oil pressure ({min_oil_press:.1f} PSI) above 4000 RPM! Bearing failure risk!")

        return {"status": status, "flags": flags}

# ------------------------------------------------------------------------------
# 3. SIDEBAR: VEHICLE PROFILE & SETUP
# ------------------------------------------------------------------------------
st.sidebar.header("1. Vehicle & Setup Profile")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key to enable AI analysis.")

make = st.sidebar.text_input("Make/Model", "Mazda Miata 1.8L")
aspiration = st.sidebar.selectbox("Aspiration", ["Turbocharged / Supercharged", "Naturally Aspirated"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["91 AKI (95 RON)", "93 AKI (98 RON)", "E85 / Flex Fuel"])
is_turbo = aspiration == "Turbocharged / Supercharged"

# ------------------------------------------------------------------------------
# 4. MAIN INTERFACE: DATALOG INPUT & EVALUATION
# ------------------------------------------------------------------------------
st.header("2. Datalog Input")
use_demo = st.checkbox("⚡ Use Demo Datalog (Instant Test)", value=False)

df = None

if use_demo:
    sample_data = {
        'RPM': [2000, 3000, 4000, 5000, 6000, 6500],
        'MAP_kPa': [40, 80, 120, 145, 155, 160],
        'AFR': [14.7, 13.5, 12.8, 12.9, 12.4, 11.8],
        'Injector_Duty': [25.0, 45.0, 68.0, 89.0, 92.5, 94.0],
        'Knock_Retard': [0.0, 0.0, 0.5, 3.2, 1.5, 0.0],
        'IAT_C': [32.0, 35.0, 42.0, 48.0, 52.0, 56.0],
        'Oil_Press_PSI': [45.0, 50.0, 42.0, 35.0, 28.0, 25.0]
    }
    df = pd.DataFrame(sample_data)
    st.success("Loaded Enhanced Demo Datalog!")
else:
    uploaded_file = st.file_uploader("Upload CSV Datalog (Supported: Megasquirt, HP Tuners, Cobb, etc.)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

if df is not None:
    tab1, tab2, tab3 = st.tabs(["🛡️ Safety Evaluation", "📊 Interactive Visualizations", "📋 Beta Audit Checklist"])

    with tab1:
        st.subheader("Raw Telemetry Preview")
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
                    st.caption(EDUCATIONAL_DISCLAIMER)

    with tab2:
        st.subheader("Interactive Telemetry Analysis")
        
        col1, col2 = st.subplots(2)
        
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
        
        - [ ] **Deterministic Safety Thresholds:** Are the IAT (> 45°C) and Oil Pressure (< 30 PSI @ 4000 RPM) limits configured correctly for this engine setup?
        - [ ] **CSV Compatibility:** Does the parser cleanly ingest datalogs from target hardware (e.g., Megasquirt, HP Tuners, Haltech)?
        - [ ] **AI Guardrail Boundaries:** Confirm the AI assistant explains thermodynamic principles without generating binary flash files.
        - [ ] **Visual Telemetry:** Are the Interactive Plotly chart boundaries clearly indicating rich vs. lean conditions under load?
        """)

# ------------------------------------------------------------------------------
# 5. FOOTER
# ------------------------------------------------------------------------------
st.divider()
st.caption("TunerAI Platform Version 0.2-Beta | Off-Road & Competition Use Only")
