import streamlit as st
import pandas as pd
import json
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
# 2. DETERMINISTIC HARDWARE FAILSAFE ENGINE (PYTHON)
# ------------------------------------------------------------------------------
class EngineSafetyGuardrail:
    """Scans datalogs for physical safety thresholds before AI analysis."""
    
    def evaluate_log(self, df: pd.DataFrame, is_turbo: bool) -> dict:
        flags = []
        status = "PASSED"

        # Rule 1: Injector Duty Cycle Check (> 85% warning, > 90% critical)
        if 'Injector_Duty' in df.columns:
            max_idc = df['Injector_Duty'].max()
            if max_idc > 90.0:
                status = "CRITICAL"
                flags.append(f"🔴 CRITICAL: Injector Duty Cycle reached {max_idc:.1f}%. Risk of fuel starvation!")
            elif max_idc > 85.0:
                if status != "CRITICAL": status = "WARNING"
                flags.append(f"🟡 WARNING: Injector Duty Cycle reached {max_idc:.1f}%. Approaching safety limit.")

        # Rule 2: Lean under boost check (MAP > 115 kPa = Boost)
        if is_turbo and 'MAP_kPa' in df.columns and 'AFR' in df.columns:
            boosted = df[df['MAP_kPa'] > 115]
            if not boosted.empty:
                max_boost_afr = boosted['AFR'].max()
                if max_boost_afr > 12.5:
                    status = "CRITICAL"
                    flags.append(f"🔴 CRITICAL: Lean AFR of {max_boost_afr:.1f}:1 detected under positive manifold pressure!")

        # Rule 3: Knock Retard Check
        if 'Knock_Retard' in df.columns:
            max_knock = df['Knock_Retard'].max()
            if max_knock > 3.0:
                status = "CRITICAL"
                flags.append(f"🔴 CRITICAL: Knock sensor pulled {max_knock:.1f}° of timing advance due to detonation.")
            elif max_knock > 1.0:
                if status != "CRITICAL": status = "WARNING"
                flags.append(f"🟡 WARNING: Minor knock detected ({max_knock:.1f}° retard).")

        return {"status": status, "flags": flags}

# ------------------------------------------------------------------------------
# 3. SIDEBAR: VEHICLE SETUP & API KEYS
# ------------------------------------------------------------------------------
st.sidebar.header("1. Vehicle & Setup Profile")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key to enable AI analysis.")

make = st.sidebar.text_input("Make/Model", "Mazda Miata 1.8L")
aspiration = st.sidebar.selectbox("Aspiration", ["Turbocharged / Supercharged", "Naturally Aspirated"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["91 AKI (95 RON)", "93 AKI (98 RON)", "E85 / Flex Fuel"])
is_turbo = aspiration == "Turbocharged / Supercharged"

# ------------------------------------------------------------------------------
# 4. MAIN INTERFACE: DATALOG UPLOAD & EVALUATION
# ------------------------------------------------------------------------------
st.header("2. Datalog Input")

use_demo = st.checkbox("⚡ Use Demo Datalog (Instant Test)", value=False)

df = None

if use_demo:
    sample_data = {
        'RPM': [2000, 3000, 4000, 5000, 6000],
        'MAP_kPa': [40, 80, 120, 145, 150],
        'AFR': [14.7, 13.5, 12.8, 12.9, 11.8],
        'Injector_Duty': [25.0, 45.0, 68.0, 89.0, 92.5],
        'Knock_Retard': [0.0, 0.0, 0.5, 3.2, 0.0]
    }
    df = pd.DataFrame(sample_data)
    st.success("Loaded Demo Datalog!")
else:
    uploaded_file = st.file_uploader("Upload CSV Datalog (Supported: Megasquirt, HP Tuners, Cobb, etc.)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

if df is not None:
    st.subheader("Raw Telemetry Preview")
    st.dataframe(df.head())

    # Step 1: Run Deterministic Safety Check
    guardrail = EngineSafetyGuardrail()
    safety_results = guardrail.evaluate_log(df, is_turbo=is_turbo)

    st.subheader("3. Deterministic Safety Evaluation")
    if safety_results["status"] == "CRITICAL":
        st.error("🔴 CRITICAL SAFETY BREACH DETECTED")
    elif safety_results["status"] == "WARNING":
        st.warning("🟡 SAFETY WARNINGS DETECTED")
    else:
        st.success("🟢 ALL DETERMINISTIC GUARDRAILS PASSED")

    for flag in safety_results["flags"]:
        st.write(flag)

    # Step 2: AI Diagnostic Analysis
    st.subheader("4. AI Educational Diagnostic & Lessons")
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

# ------------------------------------------------------------------------------
# 5. FOOTER
# ------------------------------------------------------------------------------
st.divider()
st.caption("TunerAI Platform Version 0.1-Beta | Off-Road & Competition Use Only")import streamlit as st
import pandas as pd
import json
import os
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
# 2. DETERMINISTIC HARDWARE FAILSAFE ENGINE (PYTHON)
# ------------------------------------------------------------------------------
class EngineSafetyGuardrail:
    """Scans datalogs for physical safety thresholds before AI analysis."""
    
    def evaluate_log(self, df: pd.DataFrame, is_turbo: bool) -> dict:
        flags = []
        status = "PASSED"

        # Rule 1: Injector Duty Cycle Check (> 85% warning, > 90% critical)
        if 'Injector_Duty' in df.columns:
            max_idc = df['Injector_Duty'].max()
            if max_idc > 90.0:
                status = "CRITICAL"
                flags.append(f"🔴 CRITICAL: Injector Duty Cycle reached {max_idc:.1f}%. Risk of fuel starvation!")
            elif max_idc > 85.0:
                if status != "CRITICAL": status = "WARNING"
                flags.append(f"🟡 WARNING: Injector Duty Cycle reached {max_idc:.1f}%. Approaching safety limit.")

        # Rule 2: Lean under boost check (MAP > 115 kPa = Boost)
        if is_turbo and 'MAP_kPa' in df.columns and 'AFR' in df.columns:
            boosted = df[df['MAP_kPa'] > 115]
            if not boosted.empty:
                max_boost_afr = boosted['AFR'].max()
                if max_boost_afr > 12.5:
                    status = "CRITICAL"
                    flags.append(f"🔴 CRITICAL: Lean AFR of {max_boost_afr:.1f}:1 detected under positive manifold pressure!")

        # Rule 3: Knock Retard Check
        if 'Knock_Retard' in df.columns:
            max_knock = df['Knock_Retard'].max()
            if max_knock > 3.0:
                status = "CRITICAL"
                flags.append(f"🔴 CRITICAL: Knock sensor pulled {max_knock:.1f}° of timing advance due to detonation.")
            elif max_knock > 1.0:
                if status != "CRITICAL": status = "WARNING"
                flags.append(f"🟡 WARNING: Minor knock detected ({max_knock:.1f}° retard).")

        return {"status": status, "flags": flags}

# ------------------------------------------------------------------------------
# 3. SIDEBAR: VEHICLE SETUP & API KEYS
# ------------------------------------------------------------------------------
st.sidebar.header("1. Vehicle & Setup Profile")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key to enable AI analysis.")

make = st.sidebar.text_input("Make/Model", "Mazda Miata 1.8L")
aspiration = st.sidebar.selectbox("Aspiration", ["Turbocharged / Supercharged", "Naturally Aspirated"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["91 AKI (95 RON)", "93 AKI (98 RON)", "E85 / Flex Fuel"])
is_turbo = aspiration == "Turbocharged / Supercharged"

# ------------------------------------------------------------------------------
# 4. MAIN INTERFACE: DATALOG UPLOAD & EVALUATION
# ------------------------------------------------------------------------------
st.header("2. Upload Datalog (.csv)")
uploaded_file = st.file_uploader("Upload CSV Datalog (Supported: Megasquirt, HP Tuners, Cobb, etc.)", type=["csv"])

# Demo Data Generator
if st.button("Load Sample Datalog for Testing"):
    sample_data = {
        'RPM': [2000, 3000, 4000, 5000, 6000],
        'MAP_kPa': [40, 80, 120, 145, 150],
        'AFR': [14.7, 13.5, 12.8, 12.9, 11.8],
        'Injector_Duty': [25.0, 45.0, 68.0, 89.0, 92.5],
        'Knock_Retard': [0.0, 0.0, 0.5, 3.2, 0.0]
    }
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv("sample_datalog.csv", index=False)
    st.success("Sample datalog generated! Upload 'sample_datalog.csv' above to test.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Telemetry Preview")
    st.dataframe(df.head())

    # Step 1: Run Deterministic Safety Check
    guardrail = EngineSafetyGuardrail()
    safety_results = guardrail.evaluate_log(df, is_turbo=is_turbo)

    st.subheader("3. Deterministic Safety Evaluation")
    if safety_results["status"] == "CRITICAL":
        st.error("🔴 CRITICAL SAFETY BREACH DETECTED")
    elif safety_results["status"] == "WARNING":
        st.warning("🟡 SAFETY WARNINGS DETECTED")
    else:
        st.success("🟢 ALL DETERMINISTIC GUARDRAILS PASSED")

    for flag in safety_results["flags"]:
        st.write(flag)

    # Step 2: AI Diagnostic Analysis
    st.subheader("4. AI Educational Diagnostic & Lessons")
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

# ------------------------------------------------------------------------------
# 5. FOOTER
# ------------------------------------------------------------------------------
st.divider()
st.caption("TunerAI Platform Version 0.1-Beta | Off-Road & Competition Use Only")
