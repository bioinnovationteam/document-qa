
app_code = r'''
"""
MedtechSandbox MVP
A single-file Streamlit application for lean medtech startup planning.

Run locally:
    streamlit run app.py

Deploy:
    Push app.py + requirements.txt to GitHub and deploy on Streamlit Community Cloud.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import date
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="MedtechSandbox",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Utility functions
# -----------------------------
def clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, float(value)))


def weighted_score(values: Dict[str, float], weights: Dict[str, float]) -> float:
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    return sum(values[k] * weights.get(k, 0) for k in values) / total_weight


def normalize_0_10(value: float, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return 0.0
    return clamp(10 * (value - min_value) / (max_value - min_value))


def currency(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:,.1f}M"
    if abs(x) >= 1_000:
        return f"${x/1_000:,.0f}K"
    return f"${x:,.0f}"


def download_df_button(df: pd.DataFrame, filename: str, label: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


# -----------------------------
# Data models
# -----------------------------
@dataclass
class Need:
    name: str
    clinical_problem: str
    stakeholder: str
    frequency: float
    severity: float
    current_solution_gap: float
    economic_burden: float
    reimbursement_alignment: float
    founder_advantage: float
    evidence_access: float

    def score(self) -> float:
        values = {
            "frequency": self.frequency,
            "severity": self.severity,
            "current_solution_gap": self.current_solution_gap,
            "economic_burden": self.economic_burden,
            "reimbursement_alignment": self.reimbursement_alignment,
            "founder_advantage": self.founder_advantage,
            "evidence_access": self.evidence_access,
        }
        weights = {
            "frequency": 1.1,
            "severity": 1.4,
            "current_solution_gap": 1.3,
            "economic_burden": 1.1,
            "reimbursement_alignment": 1.2,
            "founder_advantage": 1.0,
            "evidence_access": 0.9,
        }
        return weighted_score(values, weights)


@dataclass
class Concept:
    name: str
    mechanism: str
    intended_use: str
    technical_feasibility: float
    clinical_utility: float
    usability: float
    manufacturability: float
    ip_strength: float
    regulatory_clarity: float
    reimbursement_fit: float
    strategic_exit_fit: float

    def score(self) -> float:
        values = {
            "technical_feasibility": self.technical_feasibility,
            "clinical_utility": self.clinical_utility,
            "usability": self.usability,
            "manufacturability": self.manufacturability,
            "ip_strength": self.ip_strength,
            "regulatory_clarity": self.regulatory_clarity,
            "reimbursement_fit": self.reimbursement_fit,
            "strategic_exit_fit": self.strategic_exit_fit,
        }
        weights = {
            "technical_feasibility": 1.1,
            "clinical_utility": 1.4,
            "usability": 1.0,
            "manufacturability": 1.0,
            "ip_strength": 1.2,
            "regulatory_clarity": 1.2,
            "reimbursement_fit": 1.2,
            "strategic_exit_fit": 1.3,
        }
        return weighted_score(values, weights)


@dataclass
class Requirement:
    req_id: str
    user_need: str
    design_input: str
    design_output: str
    verification_test: str
    validation_activity: str
    risk_control: str
    status: str


@dataclass
class FMEAItem:
    item: str
    failure_mode: str
    effect: str
    cause: str
    severity: int
    occurrence: int
    detection: int
    mitigation: str

    def rpn(self) -> int:
        return int(self.severity * self.occurrence * self.detection)


# -----------------------------
# Default templates
# -----------------------------
DEFAULT_REQUIREMENTS = [
    Requirement(
        "REQ-001",
        "Device must address a clinically meaningful workflow pain point.",
        "The system shall produce an interpretable result for the intended user.",
        "Prototype result screen and result interpretation logic.",
        "Bench test with known positive and negative samples.",
        "Simulated-use validation with representative users.",
        "Clear result labeling and error states.",
        "Draft",
    ),
    Requirement(
        "REQ-002",
        "Device must be usable in the intended clinical or home environment.",
        "The system shall require minimal training and fit normal workflow.",
        "User interface, instructions for use, and workflow map.",
        "Formative usability test.",
        "Summative human factors validation, if required.",
        "Use-error mitigation through labeling and interface constraints.",
        "Draft",
    ),
    Requirement(
        "REQ-003",
        "Device must support regulatory submission strategy.",
        "The system shall maintain traceability from need to verification.",
        "Traceability matrix and design history file index.",
        "Internal design review.",
        "Design validation review.",
        "Document control and change history.",
        "Draft",
    ),
]

DEFAULT_FMEA = [
    FMEAItem(
        "Sensor / assay",
        "False negative",
        "Missed disease or delayed intervention",
        "Low sensitivity, user error, sample degradation",
        9,
        4,
        4,
        "Improve analytical sensitivity, add controls, validate sample handling.",
    ),
    FMEAItem(
        "Sensor / assay",
        "False positive",
        "Unnecessary follow-up, anxiety, cost",
        "Cross-reactivity, calibration drift, contamination",
        7,
        4,
        5,
        "Specificity testing, calibration checks, confirmatory workflow.",
    ),
    FMEAItem(
        "Software",
        "Incorrect result interpretation",
        "Wrong clinical action",
        "Algorithm bug, threshold error, poor UI",
        8,
        3,
        4,
        "Locked thresholds, unit tests, clinical review, verification dataset.",
    ),
    FMEAItem(
        "Workflow",
        "Result not documented",
        "Loss of clinical continuity",
        "No EHR integration or manual transcription error",
        6,
        5,
        5,
        "FHIR/HL7 workflow, exportable report, audit trail.",
    ),
]


# -----------------------------
# Session state
# -----------------------------
if "needs" not in st.session_state:
    st.session_state.needs = []

if "concepts" not in st.session_state:
    st.session_state.concepts = []

if "requirements" not in st.session_state:
    st.session_state.requirements = [asdict(x) for x in DEFAULT_REQUIREMENTS]

if "fmea" not in st.session_state:
    st.session_state.fmea = [asdict(x) for x in DEFAULT_FMEA]


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🧬 MedtechSandbox")
st.sidebar.caption("Lean medtech venture design system")

module = st.sidebar.radio(
    "Module",
    [
        "Dashboard",
        "Needs Finding",
        "Concept Generation",
        "Concept Screening",
        "Patent Search Planner",
        "Prototype & Simulation",
        "Clinical Integration",
        "Traceability / DHF",
        "FMEA",
        "FDA Strategy",
        "Reimbursement Strategy",
        "Business & Finance",
        "Supply Chain & Sales",
        "Exit Strategy",
        "Export Workspace",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Scoring scale: 0 = poor, 10 = excellent")


# -----------------------------
# Dashboard
# -----------------------------
if module == "Dashboard":
    st.title("MedtechSandbox")
    st.subheader("A venture operating system for lean medtech teams")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Needs", len(st.session_state.needs))
    col2.metric("Concepts", len(st.session_state.concepts))
    col3.metric("Requirements", len(st.session_state.requirements))
    col4.metric("FMEA Items", len(st.session_state.fmea))

    st.markdown(
        """
        This MVP helps a lean medtech startup connect:
        **clinical needs → concepts → prototypes → risk controls → FDA strategy → reimbursement → business model → exit thesis**.

        It is not a substitute for legal, regulatory, reimbursement, clinical, or patent counsel.
        Use it as an internal strategy, documentation, and decision-support system.
        """
    )

    st.divider()

    if st.session_state.needs:
        needs_df = pd.DataFrame(st.session_state.needs)
        needs_df["score"] = needs_df.apply(lambda r: Need(**r).score(), axis=1)
        st.subheader("Top Needs")
        st.dataframe(needs_df.sort_values("score", ascending=False), use_container_width=True)

    if st.session_state.concepts:
        concepts_df = pd.DataFrame(st.session_state.concepts)
        concepts_df["score"] = concepts_df.apply(lambda r: Concept(**r).score(), axis=1)
        st.subheader("Top Concepts")
        st.dataframe(concepts_df.sort_values("score", ascending=False), use_container_width=True)

    st.subheader("Recommended Next Action")
    if not st.session_state.needs:
        st.info("Start with Needs Finding. Add at least 3 needs before inventing solutions.")
    elif not st.session_state.concepts:
        st.info("Move to Concept Generation. Generate at least 3 concepts for the highest-scoring need.")
    elif len(st.session_state.requirements) < 5:
        st.info("Expand the Traceability / DHF module. Add measurable design inputs and verification tests.")
    else:
        st.success("Proceed to FDA Strategy and Business & Finance. Quantify the path to de-risking and exit.")


# -----------------------------
# Needs Finding
# -----------------------------
elif module == "Needs Finding":
    st.title("Needs Finding & Screening")

    with st.form("need_form", clear_on_submit=True):
        name = st.text_input("Need name", "At-home quantitative kidney screening")
        clinical_problem = st.text_area("Clinical problem", "Patients at risk for CKD are under-screened because urine testing is inconvenient and poorly completed.")
        stakeholder = st.selectbox("Primary stakeholder", ["Patient", "Clinician", "Nurse", "Hospital", "Payor", "Lab", "Care manager", "Other"])

        c1, c2, c3 = st.columns(3)
        frequency = c1.slider("Frequency", 0.0, 10.0, 7.0)
        severity = c2.slider("Severity", 0.0, 10.0, 8.0)
        current_solution_gap = c3.slider("Current solution gap", 0.0, 10.0, 7.0)

        c4, c5, c6, c7 = st.columns(4)
        economic_burden = c4.slider("Economic burden", 0.0, 10.0, 8.0)
        reimbursement_alignment = c5.slider("Reimbursement alignment", 0.0, 10.0, 7.0)
        founder_advantage = c6.slider("Founder/team advantage", 0.0, 10.0, 6.0)
        evidence_access = c7.slider("Evidence access", 0.0, 10.0, 6.0)

        submitted = st.form_submit_button("Add Need")

    if submitted:
        need = Need(
            name,
            clinical_problem,
            stakeholder,
            frequency,
            severity,
            current_solution_gap,
            economic_burden,
            reimbursement_alignment,
            founder_advantage,
            evidence_access,
        )
        st.session_state.needs.append(asdict(need))
        st.success(f"Added need: {name}")

    if st.session_state.needs:
        df = pd.DataFrame(st.session_state.needs)
        df["score"] = df.apply(lambda r: Need(**r).score(), axis=1)
        df = df.sort_values("score", ascending=False)
        st.subheader("Need Ranking")
        st.dataframe(df, use_container_width=True)
        download_df_button(df, "needs_screening.csv", "Download needs CSV")


# -----------------------------
# Concept Generation
# -----------------------------
elif module == "Concept Generation":
    st.title("Concept Generation")

    selected_need = None
    if st.session_state.needs:
        need_names = [n["name"] for n in st.session_state.needs]
        selected_need = st.selectbox("Generate concepts for need", need_names)
    else:
        st.warning("Add at least one need first.")

    st.subheader("Structured concept prompts")
    st.markdown(
        """
        Use these prompts to generate concepts:
        - Replace the current workflow step with automation.
        - Move the diagnostic or treatment closer to the patient.
        - Convert qualitative assessment into quantitative measurement.
        - Reduce training burden.
        - Turn a fragmented workflow into a single closed-loop system.
        - Create a disposable + reusable platform split.
        - Make the intervention safer in low-resource or high-throughput settings.
        """
    )

    with st.form("concept_form", clear_on_submit=True):
        name = st.text_input("Concept name", "Smartphone-enabled quantitative test")
        mechanism = st.text_area("Mechanism", "Colorimetric assay imaged by smartphone and interpreted by software.")
        intended_use = st.text_area("Intended use", "Screening and monitoring of at-risk patients outside the clinic.")

        c1, c2, c3, c4 = st.columns(4)
        technical_feasibility = c1.slider("Technical feasibility", 0.0, 10.0, 7.0)
        clinical_utility = c2.slider("Clinical utility", 0.0, 10.0, 8.0)
        usability = c3.slider("Usability", 0.0, 10.0, 7.0)
        manufacturability = c4.slider("Manufacturability", 0.0, 10.0, 6.0)

        c5, c6, c7, c8 = st.columns(4)
        ip_strength = c5.slider("IP strength", 0.0, 10.0, 6.0)
        regulatory_clarity = c6.slider("Regulatory clarity", 0.0, 10.0, 6.0)
        reimbursement_fit = c7.slider("Reimbursement fit", 0.0, 10.0, 7.0)
        strategic_exit_fit = c8.slider("Strategic exit fit", 0.0, 10.0, 7.0)

        submitted = st.form_submit_button("Add Concept")

    if submitted:
        concept = Concept(
            name,
            mechanism,
            intended_use,
            technical_feasibility,
            clinical_utility,
            usability,
            manufacturability,
            ip_strength,
            regulatory_clarity,
            reimbursement_fit,
            strategic_exit_fit,
        )
        st.session_state.concepts.append(asdict(concept))
        st.success(f"Added concept: {name}")


# -----------------------------
# Concept Screening
# -----------------------------
elif module == "Concept Screening":
    st.title("Concept Screening")

    if not st.session_state.concepts:
        st.warning("Add concepts first.")
    else:
        df = pd.DataFrame(st.session_state.concepts)
        df["score"] = df.apply(lambda r: Concept(**r).score(), axis=1)
        df = df.sort_values("score", ascending=False)

        st.dataframe(df, use_container_width=True)
        download_df_button(df, "concept_screening.csv", "Download concept screening CSV")

        st.subheader("Score Map")
        numeric_cols = [
            "technical_feasibility",
            "clinical_utility",
            "usability",
            "manufacturability",
            "ip_strength",
            "regulatory_clarity",
            "reimbursement_fit",
            "strategic_exit_fit",
        ]
        st.bar_chart(df.set_index("name")[numeric_cols])


# -----------------------------
# Patent Search Planner
# -----------------------------
elif module == "Patent Search Planner":
    st.title("Patent Search Planner")

    st.markdown(
        """
        This module does not provide a legal freedom-to-operate opinion.
        It creates a structured patent-search plan for founders before speaking with patent counsel.
        """
    )

    device_terms = st.text_input("Device / technology terms", "urine albumin creatinine ratio smartphone colorimetric assay")
    companies = st.text_input("Known companies / assignees", "Healthy.io, Siemens Healthineers, Abbott, Roche, Labcorp")
    mechanisms = st.text_input("Mechanisms to search", "dipstick, lateral flow, colorimetric nanoparticle, image analysis, calibration card")

    queries = [
        f'"{device_terms}" patent',
        f'({device_terms}) ({mechanisms}) patent claims',
        f'({companies}) ({device_terms}) patent',
        f'("method of detecting" OR "system for detecting") ({mechanisms})',
        f'("freedom to operate" OR FTO) ({device_terms})',
    ]

    st.subheader("Search Queries")
    st.code("\n".join(queries))

    checklist = pd.DataFrame(
        {
            "Task": [
                "Search Google Patents",
                "Search USPTO Patent Center",
                "Search WIPO Patentscope",
                "Identify active independent claims",
                "Map claim elements against your concept",
                "Identify design-around options",
                "Review expiration, priority, and continuations",
                "Consult patent counsel before disclosure or fundraising",
            ],
            "Status": ["Not started"] * 8,
        }
    )
    st.dataframe(checklist, use_container_width=True)
    download_df_button(checklist, "patent_search_checklist.csv", "Download patent checklist")


# -----------------------------
# Prototype & Simulation
# -----------------------------
elif module == "Prototype & Simulation":
    st.title("Prototype & Simulation")

    st.subheader("Simple diagnostic performance simulator")

    c1, c2, c3 = st.columns(3)
    prevalence = c1.slider("Disease prevalence in tested population", 0.001, 0.50, 0.10)
    sensitivity = c2.slider("Sensitivity", 0.50, 1.00, 0.90)
    specificity = c3.slider("Specificity", 0.50, 1.00, 0.85)

    n = st.number_input("Tested population size", min_value=100, max_value=10_000_000, value=10_000, step=100)

    true_disease = n * prevalence
    no_disease = n - true_disease
    tp = true_disease * sensitivity
    fn = true_disease - tp
    tn = no_disease * specificity
    fp = no_disease - tn

    ppv = tp / (tp + fp) if tp + fp > 0 else 0
    npv = tn / (tn + fn) if tn + fn > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("True positives", f"{tp:,.0f}")
    col2.metric("False negatives", f"{fn:,.0f}")
    col3.metric("False positives", f"{fp:,.0f}")
    col4.metric("True negatives", f"{tn:,.0f}")

    col5, col6 = st.columns(2)
    col5.metric("PPV", f"{ppv:.1%}")
    col6.metric("NPV", f"{npv:.1%}")

    cm = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["Disease present", "Disease absent"],
        columns=["Test positive", "Test negative"],
    )
    st.dataframe(cm.style.format("{:,.0f}"), use_container_width=True)

    st.subheader("Prototype decision")
    if sensitivity < 0.8:
        st.error("Sensitivity is likely too low for screening unless confirmatory workflow is strong.")
    elif specificity < 0.8:
        st.warning("Specificity may cause adoption problems due to false positives.")
    else:
        st.success("Performance assumptions are plausible enough for early prototype planning.")


# -----------------------------
# Clinical Integration
# -----------------------------
elif module == "Clinical Integration":
    st.title("Clinical Integration")

    st.subheader("Workflow mapping")
    workflow = pd.DataFrame(
        {
            "Step": [
                "Patient identified",
                "Test ordered",
                "Sample collected",
                "Device/procedure performed",
                "Result generated",
                "Result documented",
                "Clinical action taken",
                "Follow-up completed",
            ],
            "Owner": [
                "Clinician / care gap team",
                "Clinician",
                "Patient / nurse",
                "Patient / staff",
                "Device/software",
                "EHR / care team",
                "Clinician",
                "Care manager",
            ],
            "Failure risk": [
                "Patient missed",
                "Order not placed",
                "Sample error",
                "Use error",
                "Algorithm/device error",
                "Documentation gap",
                "No action",
                "No follow-up",
            ],
        }
    )
    st.dataframe(workflow, use_container_width=True)

    st.subheader("Integration checklist")
    integration = pd.DataFrame(
        {
            "Integration Area": [
                "EHR result mapping",
                "LOINC / coding",
                "FHIR or HL7 interface",
                "Patient instructions",
                "Clinician result view",
                "Audit trail",
                "Data privacy",
                "Escalation workflow",
            ],
            "Question": [
                "Where does the result appear?",
                "What code represents the measurement?",
                "How does the result move?",
                "Can users complete the task safely?",
                "Can clinicians interpret quickly?",
                "Can actions be reconstructed?",
                "Is PHI protected?",
                "Who acts on abnormal results?",
            ],
        }
    )
    st.dataframe(integration, use_container_width=True)
    download_df_button(integration, "clinical_integration_checklist.csv", "Download integration checklist")


# -----------------------------
# Traceability / DHF
# -----------------------------
elif module == "Traceability / DHF":
    st.title("Traceability Matrix / DHF Builder")

    with st.form("req_form", clear_on_submit=True):
        req_id = st.text_input("Requirement ID", f"REQ-{len(st.session_state.requirements)+1:03d}")
        user_need = st.text_area("User need")
        design_input = st.text_area("Design input")
        design_output = st.text_area("Design output")
        verification_test = st.text_area("Verification test")
        validation_activity = st.text_area("Validation activity")
        risk_control = st.text_area("Risk control")
        status = st.selectbox("Status", ["Draft", "In review", "Approved", "Verified", "Validated"])
        submitted = st.form_submit_button("Add Requirement")

    if submitted:
        req = Requirement(req_id, user_need, design_input, design_output, verification_test, validation_activity, risk_control, status)
        st.session_state.requirements.append(asdict(req))
        st.success(f"Added {req_id}")

    df = pd.DataFrame(st.session_state.requirements)
    st.dataframe(df, use_container_width=True)
    download_df_button(df, "traceability_matrix.csv", "Download traceability matrix CSV")

    st.subheader("DHF Index")
    dhf = pd.DataFrame(
        {
            "DHF Section": [
                "User needs",
                "Design inputs",
                "Design outputs",
                "Design reviews",
                "Verification",
                "Validation",
                "Risk management",
                "Change history",
                "Regulatory strategy",
            ],
            "Evidence": [
                "Needs statements, VOC, stakeholder interviews",
                "Measurable requirements",
                "Drawings, software, prototypes, specifications",
                "Review minutes and approvals",
                "Bench, analytical, software, and usability tests",
                "Clinical, simulated-use, or summative validation",
                "FMEA, hazards, mitigations, residual risk",
                "Version history and design changes",
                "Predicate, Q-sub, submission plan",
            ],
        }
    )
    st.dataframe(dhf, use_container_width=True)


# -----------------------------
# FMEA
# -----------------------------
elif module == "FMEA":
    st.title("FMEA / Risk Management")

    with st.form("fmea_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        item = c1.text_input("Item / subsystem")
        failure_mode = c2.text_input("Failure mode")
        effect = st.text_area("Effect")
        cause = st.text_area("Cause")
        c3, c4, c5 = st.columns(3)
        severity = c3.slider("Severity", 1, 10, 7)
        occurrence = c4.slider("Occurrence", 1, 10, 4)
        detection = c5.slider("Detection", 1, 10, 4)
        mitigation = st.text_area("Mitigation")
        submitted = st.form_submit_button("Add FMEA Item")

    if submitted:
        item_obj = FMEAItem(item, failure_mode, effect, cause, severity, occurrence, detection, mitigation)
        st.session_state.fmea.append(asdict(item_obj))
        st.success("Added FMEA item")

    df = pd.DataFrame(st.session_state.fmea)
    df["RPN"] = df.apply(lambda r: FMEAItem(**{k: r[k] for k in FMEAItem.__annotations__.keys()}).rpn(), axis=1)
    df = df.sort_values("RPN", ascending=False)
    st.dataframe(df, use_container_width=True)
    download_df_button(df, "fmea.csv", "Download FMEA CSV")

    high = df[df["RPN"] >= 200]
    if len(high):
        st.error(f"{len(high)} high-priority risks need mitigation before design freeze.")
    else:
        st.success("No RPN values above 200 in the current FMEA.")


# -----------------------------
# FDA Strategy
# -----------------------------
elif module == "FDA Strategy":
    st.title("FDA Strategy")

    c1, c2, c3 = st.columns(3)
    invasive = c1.selectbox("Is the device invasive?", ["No", "Yes"])
    life_supporting = c2.selectbox("Life-supporting / life-sustaining?", ["No", "Yes"])
    existing_predicate = c3.selectbox("Likely predicate exists?", ["Yes", "No", "Unknown"])

    diagnostic = st.selectbox("Is it diagnostic / measuring clinical analyte?", ["No", "Yes"])
    software_drives_decision = st.selectbox("Does software drive clinical decision?", ["No", "Yes"])
    novel_technology = st.selectbox("Novel technology or new intended use?", ["No", "Yes", "Unknown"])

    risk_points = 0
    risk_points += 3 if invasive == "Yes" else 0
    risk_points += 5 if life_supporting == "Yes" else 0
    risk_points += 2 if diagnostic == "Yes" else 0
    risk_points += 2 if software_drives_decision == "Yes" else 0
    risk_points += 3 if existing_predicate == "No" else 1 if existing_predicate == "Unknown" else 0
    risk_points += 2 if novel_technology != "No" else 0

    if life_supporting == "Yes" or risk_points >= 9:
        pathway = "Potential PMA or high-burden De Novo. Get regulatory counsel immediately."
    elif existing_predicate == "Yes" and risk_points <= 6:
        pathway = "Likely 510(k) candidate, subject to intended use and predicate analysis."
    elif existing_predicate in ["No", "Unknown"]:
        pathway = "Possible De Novo or 510(k) with careful predicate strategy. FDA Q-sub recommended."
    else:
        pathway = "Moderate-risk strategy. FDA Q-sub recommended."

    st.subheader("Preliminary pathway estimate")
    st.info(pathway)

    testing = pd.DataFrame(
        {
            "Testing Area": [
                "Bench performance",
                "Analytical validation",
                "Software verification",
                "Cybersecurity",
                "Electrical safety / EMC",
                "Biocompatibility",
                "Human factors",
                "Clinical validation",
                "Labeling",
            ],
            "Likely Relevance": [
                "High",
                "High" if diagnostic == "Yes" else "Medium",
                "High" if software_drives_decision == "Yes" else "Medium",
                "Medium" if software_drives_decision == "Yes" else "Low",
                "Depends on electronics",
                "Depends on patient contact",
                "High",
                "Depends on claims",
                "High",
            ],
        }
    )
    st.dataframe(testing, use_container_width=True)
    download_df_button(testing, "fda_testing_plan.csv", "Download FDA testing plan")


# -----------------------------
# Reimbursement Strategy
# -----------------------------
elif module == "Reimbursement Strategy":
    st.title("Reimbursement Strategy")

    c1, c2, c3 = st.columns(3)
    payment_route = c1.selectbox("Primary payment route", ["Existing CPT / HCPCS", "New code needed", "Value-based contract", "Direct hospital budget", "Self-pay"])
    buyer = c2.selectbox("Economic buyer", ["Payor", "Hospital", "Clinic", "Lab", "Employer", "Patient"])
    metric = c3.text_input("Value metric", "care gap closure, avoided disease progression, reduced readmissions")

    st.subheader("Reimbursement logic")
    if payment_route == "Existing CPT / HCPCS":
        st.success("Best-case path: align product workflow with existing billing and documentation requirements.")
    elif payment_route == "Value-based contract":
        st.info("Strong startup path if you can quantify savings and improve quality metrics.")
    elif payment_route == "New code needed":
        st.warning("Longer timeline. Build evidence for clinical utility and economic value early.")
    else:
        st.info("Validate willingness to pay directly with the budget owner.")

    roi = pd.DataFrame(
        {
            "Evidence Needed": [
                "Clinical utility",
                "Economic model",
                "Workflow improvement",
                "Coding/billing fit",
                "Quality metric impact",
                "Budget impact",
            ],
            "Founder Task": [
                "Show the result changes care decisions.",
                "Estimate avoided costs or revenue capture.",
                "Prove time, adherence, or completion improvement.",
                "Map to existing codes or payment pathway.",
                "Tie to HEDIS, STAR, MIPS, ACO, or institutional KPI if applicable.",
                "Build per-member, per-patient, or per-site ROI model.",
            ],
        }
    )
    st.dataframe(roi, use_container_width=True)
    download_df_button(roi, "reimbursement_strategy.csv", "Download reimbursement strategy")


# -----------------------------
# Business & Finance
# -----------------------------
elif module == "Business & Finance":
    st.title("Business & Financial Planning")

    c1, c2, c3 = st.columns(3)
    addressable_patients = c1.number_input("Addressable patients / year", min_value=0, value=1_000_000, step=10_000)
    price = c2.number_input("Revenue per use / patient", min_value=0.0, value=50.0, step=1.0)
    gross_margin = c3.slider("Gross margin", 0.0, 1.0, 0.65)

    c4, c5, c6 = st.columns(3)
    penetration = c4.slider("Market penetration", 0.0, 1.0, 0.02)
    annual_fixed_cost = c5.number_input("Annual fixed operating cost", min_value=0.0, value=750_000.0, step=50_000.0)
    development_cost = c6.number_input("Remaining development / regulatory cost", min_value=0.0, value=2_500_000.0, step=100_000.0)

    revenue = addressable_patients * penetration * price
    gross_profit = revenue * gross_margin
    operating_profit = gross_profit - annual_fixed_cost

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual revenue", currency(revenue))
    col2.metric("Gross profit", currency(gross_profit))
    col3.metric("Operating profit", currency(operating_profit))
    col4.metric("Capital need", currency(development_cost))

    st.subheader("5-year simple projection")
    years = np.arange(1, 6)
    penetration_curve = np.minimum(penetration * years * 0.8, 0.25)
    rev_curve = addressable_patients * penetration_curve * price
    profit_curve = rev_curve * gross_margin - annual_fixed_cost
    projection = pd.DataFrame(
        {
            "Year": years,
            "Penetration": penetration_curve,
            "Revenue": rev_curve,
            "Operating Profit": profit_curve,
        }
    )
    st.dataframe(projection.style.format({"Penetration": "{:.1%}", "Revenue": "${:,.0f}", "Operating Profit": "${:,.0f}"}), use_container_width=True)
    st.line_chart(projection.set_index("Year")[["Revenue", "Operating Profit"]])
    download_df_button(projection, "financial_projection.csv", "Download projection CSV")


# -----------------------------
# Supply Chain & Sales
# -----------------------------
elif module == "Supply Chain & Sales":
    st.title("Supply Chain & Sales Logistics")

    st.subheader("COGS model")
    c1, c2, c3, c4 = st.columns(4)
    bom = c1.number_input("Bill of materials / unit", min_value=0.0, value=8.0, step=0.5)
    assembly = c2.number_input("Assembly / unit", min_value=0.0, value=3.0, step=0.5)
    qa = c3.number_input("QA / unit", min_value=0.0, value=2.0, step=0.5)
    logistics = c4.number_input("Logistics / unit", min_value=0.0, value=2.0, step=0.5)

    cogs = bom + assembly + qa + logistics
    target_price = st.number_input("Target price / unit", min_value=0.0, value=50.0, step=1.0)
    margin = (target_price - cogs) / target_price if target_price > 0 else 0

    col1, col2 = st.columns(2)
    col1.metric("COGS / unit", currency(cogs))
    col2.metric("Gross margin", f"{margin:.1%}")

    st.subheader("Channel comparison")
    channels = pd.DataFrame(
        {
            "Channel": ["Direct sales", "Distributor", "Strategic licensee", "Health-system pilot", "Payor contract"],
            "Pros": [
                "High control and learning",
                "Faster reach",
                "Best for exit-driven scale",
                "Strong clinical evidence",
                "Strong economic alignment",
            ],
            "Cons": [
                "Expensive sales team",
                "Lower margin and less control",
                "Requires de-risked asset",
                "Slow procurement",
                "Long contracting cycle",
            ],
            "Best Use": [
                "Early enterprise learning",
                "Commodity-like devices",
                "Manufacturing and commercial scale",
                "Clinical validation",
                "Screening/adherence/value-based care",
            ],
        }
    )
    st.dataframe(channels, use_container_width=True)


# -----------------------------
# Exit Strategy
# -----------------------------
elif module == "Exit Strategy":
    st.title("Exit Strategy Optimizer")

    st.subheader("Acquisition readiness")
    c1, c2, c3, c4 = st.columns(4)
    clinical_evidence = c1.slider("Clinical evidence", 0.0, 10.0, 4.0)
    ip_position = c2.slider("IP position", 0.0, 10.0, 5.0)
    regulatory_derisking = c3.slider("Regulatory de-risking", 0.0, 10.0, 4.0)
    strategic_fit = c4.slider("Strategic acquirer fit", 0.0, 10.0, 6.0)

    c5, c6, c7, c8 = st.columns(4)
    reimbursement_evidence = c5.slider("Reimbursement evidence", 0.0, 10.0, 4.0)
    prototype_maturity = c6.slider("Prototype maturity", 0.0, 10.0, 5.0)
    manufacturing_plan = c7.slider("Manufacturing plan", 0.0, 10.0, 3.0)
    traction = c8.slider("LOIs / pilots / traction", 0.0, 10.0, 3.0)

    readiness_values = {
        "clinical_evidence": clinical_evidence,
        "ip_position": ip_position,
        "regulatory_derisking": regulatory_derisking,
        "strategic_fit": strategic_fit,
        "reimbursement_evidence": reimbursement_evidence,
        "prototype_maturity": prototype_maturity,
        "manufacturing_plan": manufacturing_plan,
        "traction": traction,
    }
    readiness_weights = {
        "clinical_evidence": 1.4,
        "ip_position": 1.3,
        "regulatory_derisking": 1.4,
        "strategic_fit": 1.3,
        "reimbursement_evidence": 1.2,
        "prototype_maturity": 1.0,
        "manufacturing_plan": 0.9,
        "traction": 1.2,
    }

    readiness = weighted_score(readiness_values, readiness_weights)
    acquisition_probability = 1 / (1 + math.exp(-(readiness - 6.5)))
    base_valuation = st.number_input("Base valuation if fully de-risked", min_value=0.0, value=40_000_000.0, step=1_000_000.0)
    expected_exit_value = acquisition_probability * base_valuation

    col1, col2, col3 = st.columns(3)
    col1.metric("Exit readiness", f"{readiness:.1f}/10")
    col2.metric("Estimated acquisition probability", f"{acquisition_probability:.1%}")
    col3.metric("Expected exit value", currency(expected_exit_value))

    priorities = sorted(readiness_values.items(), key=lambda x: x[1])
    st.subheader("Highest-leverage next de-risking priorities")
    for k, v in priorities[:3]:
        st.write(f"**{k.replace('_', ' ').title()}**: current score {v:.1f}/10")

    acquirer_df = pd.DataFrame(
        {
            "Acquirer Type": [
                "Large diagnostics company",
                "Medical device company",
                "Clinical lab / testing company",
                "Digital health platform",
                "Health system / payor innovation arm",
            ],
            "What They Buy": [
                "Validated assay, regulatory path, manufacturing plan",
                "Protected device platform and clinical workflow",
                "Test volume, reimbursement, lab integration",
                "Software workflow, data asset, patient engagement",
                "Cost savings, quality metrics, care-gap closure",
            ],
        }
    )
    st.dataframe(acquirer_df, use_container_width=True)


# -----------------------------
# Export Workspace
# -----------------------------
elif module == "Export Workspace":
    st.title("Export Workspace")

    workspace = {
        "export_date": str(date.today()),
        "needs": st.session_state.needs,
        "concepts": st.session_state.concepts,
        "requirements": st.session_state.requirements,
        "fmea": st.session_state.fmea,
    }

    st.download_button(
        "Download workspace JSON",
        data=json.dumps(workspace, indent=2),
        file_name="medtechsandbox_workspace.json",
        mime="application/json",
    )

    st.subheader("Workspace Preview")
    st.json(workspace)
'''

requirements = """streamlit>=1.33
pandas>=2.0
numpy>=1.24
"""


print("Created /mnt/data/app.py and /mnt/data/requirements.txt")
