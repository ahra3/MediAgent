"""MediAgent — Clinical Decision Support Dashboard (Streamlit).

Run:  uv run streamlit run app.py
"""

import streamlit as st
import time
from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MediAgent — Clinical Decision Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Pipeline (cached across reruns) ──────────────────────────────────────────
@st.cache_resource
def get_pipeline():
    from langgraph.checkpoint.memory import MemorySaver
    from src.graph.workflow import build_graph
    graph = build_graph()
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


pipeline = get_pipeline()


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #edf3ff;
}
.stApp {
    background: #07111f;
    background-image:
        radial-gradient(circle at top left, rgba(6,182,212,0.12), transparent 35%),
        radial-gradient(circle at bottom right, rgba(56,189,248,0.10), transparent 30%),
        linear-gradient(180deg, #07111f 0%, #0a1729 100%);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1280px; }

/* ── Hero ── */
.hero { text-align: center; padding: 3rem 0 2rem; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.25em; text-transform: uppercase; color: #38bdf8;
    margin-bottom: 0.8rem; opacity: 0.9;
}
.hero h1 {
    font-family: 'Syne', sans-serif; font-size: clamp(2.6rem, 5.5vw, 4.5rem);
    font-weight: 800; line-height: 1.05; letter-spacing: -0.03em;
    color: #f8fbff; margin: 0 0 0.8rem;
}
.hero h1 span {
    background: linear-gradient(135deg, #38bdf8, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1rem; font-weight: 300; color: #b5c3d9; max-width: 560px;
    margin: 0 auto; line-height: 1.65;
}
.hero-badges { display: flex; justify-content: center; gap: 10px; margin-top: 1.2rem; flex-wrap: wrap; }
.hero-badge {
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px; padding: 4px 14px; font-size: 0.7rem; font-weight: 500;
    font-family: 'DM Mono', monospace; letter-spacing: 0.04em; color: #b5c3d9;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), transparent);
    margin: 2rem 0;
}

/* ── Input card ── */
.input-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(56,189,248,0.18);
    border-radius: 22px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    backdrop-filter: blur(14px); box-shadow: 0 10px 40px rgba(0,0,0,0.28);
}
.stTextArea > label, .stTextInput > label {
    font-family: 'DM Mono', monospace !important; font-size: 0.72rem !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important;
    color: #38bdf8 !important; font-weight: 500 !important;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 14px !important; color: #f8fbff !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.92rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.12) !important;
}
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    border-radius: 12px !important; color: #f8fbff !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 0.95rem !important; letter-spacing: 0.04em !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.8rem 2.2rem !important; cursor: pointer !important;
    transition: all 0.2s ease !important; width: 100%;
}
.analyze-btn button {
    background: linear-gradient(135deg, #38bdf8 0%, #06b6d4 100%) !important;
    color: white !important;
    box-shadow: 0 8px 30px rgba(56,189,248,0.25) !important;
}
.analyze-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 35px rgba(56,189,248,0.35) !important;
}
.approve-btn button {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    color: white !important;
    box-shadow: 0 6px 24px rgba(34,197,94,0.25) !important;
}
.reject-btn button {
    background: transparent !important;
    border: 2px solid rgba(251,191,36,0.5) !important;
    color: #fbbf24 !important;
}

/* ── Step cards ── */
.step-card {
    background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
    position: relative; overflow: hidden; transition: all 0.25s ease;
    backdrop-filter: blur(10px);
}
.step-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 4px; border-radius: 16px 0 0 16px; background: rgba(255,255,255,0.06);
    transition: background 0.3s;
}
.step-card.active { border-color: rgba(56,189,248,0.4); background: rgba(56,189,248,0.06); }
.step-card.active::before { background: #38bdf8; }
.step-card.done { border-color: rgba(34,197,94,0.25); background: rgba(34,197,94,0.04); }
.step-card.done::before { background: #22c55e; }
.step-header { display: flex; align-items: center; gap: 0.7rem; }
.step-num {
    font-family: 'DM Mono', monospace; font-size: 0.65rem; font-weight: 500;
    letter-spacing: 0.15em; color: #38bdf8; opacity: 0.8;
}
.step-title { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: #f8fbff; }
.step-status { margin-left: auto; font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 0.1em; }
.status-waiting { color: #475569; }
.status-running { color: #38bdf8; }
.status-done { color: #22c55e; }
.step-desc { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; margin-left: 2.8rem; }

/* ── HITL review card ── */
.hitl-card {
    background: rgba(251,191,36,0.06); border: 1px solid rgba(251,191,36,0.3);
    border-radius: 20px; padding: 2rem 2.5rem; margin: 1.5rem 0;
    backdrop-filter: blur(14px); box-shadow: 0 8px 32px rgba(251,191,36,0.08);
}
.hitl-label {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.2em; text-transform: uppercase; color: #fbbf24;
    margin-bottom: 1rem; padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(251,191,36,0.2);
}
.hitl-content { font-size: 0.88rem; line-height: 1.7; color: #d8e2f0; white-space: pre-wrap; }

/* ── Report panel ── */
.report-panel {
    background: rgba(56,189,248,0.04); border: 1px solid rgba(56,189,248,0.2);
    border-radius: 22px; padding: 2.5rem 3rem; margin-top: 1.5rem;
    backdrop-filter: blur(14px);
}
.report-panel h2 {
    font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700;
    color: #f8fbff; margin-bottom: 0.5rem;
}
.report-panel h3 {
    font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700;
    color: #38bdf8; margin-top: 1.5rem; margin-bottom: 0.6rem;
}
.report-panel p, .report-panel li {
    font-size: 0.9rem; line-height: 1.75; color: #d8e2f0;
}
.report-panel strong { color: #f8fbff; }
.report-panel blockquote {
    border-left: 3px solid rgba(56,189,248,0.4); padding-left: 1rem;
    margin: 0.8rem 0; background: rgba(255,255,255,0.02); border-radius: 0 10px 10px 0;
    padding: 0.8rem 1.2rem;
}
.report-panel code {
    background: rgba(56,189,248,0.1); color: #38bdf8; padding: 2px 6px;
    border-radius: 4px; font-size: 0.78rem; font-family: 'DM Mono', monospace;
}
.risk-badge {
    display: inline-block; padding: 6px 18px; border-radius: 20px;
    font-weight: 600; font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 0.06em; font-family: 'DM Mono', monospace;
}
.risk-low { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.risk-moderate { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.risk-high { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.risk-critical { background: rgba(239,68,68,0.3); color: #fca5a5; border: 1px solid rgba(239,68,68,0.5); }

/* ── Diagnostics expander ── */
details {
    background: rgba(255,255,255,0.02); border-radius: 14px;
    padding: 0.3rem 0.8rem; border: 1px solid rgba(255,255,255,0.06);
}
details summary {
    font-family: 'DM Mono', monospace !important; font-size: 0.72rem !important;
    color: #64748b !important; letter-spacing: 0.1em !important;
}
.section-heading {
    font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 700;
    color: #f8fbff; margin: 1.5rem 0 1rem;
}
.notice {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #475569;
    text-align: center; margin-top: 3rem; letter-spacing: 0.06em; line-height: 1.8;
}
.stSpinner > div { color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def step_card(num: str, title: str, state: str, desc: str = ""):
    status_map = {
        "waiting": ("WAITING", "status-waiting", ""),
        "running": ("● RUNNING", "status-running", "active"),
        "done": ("✓ DONE", "status-done", "done"),
    }
    label, cls, card_cls = status_map.get(state, ("", "", ""))
    st.markdown(f"""
    <div class="step-card {card_cls}">
        <div class="step-header">
            <span class="step-num">{num}</span>
            <span class="step-title">{title}</span>
            <span class="step-status {cls}">{label}</span>
        </div>
        <div class="step-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def format_report(report) -> str:
    risk = report.risk_level.lower()
    risk_map = {
        "low": ("🟢", "risk-low"), "moderate": ("🟡", "risk-moderate"),
        "high": ("🔴", "risk-high"), "critical": ("⛔", "risk-critical"),
    }
    icon, css = risk_map.get(risk, ("⚪", "risk-moderate"))

    md = f'<span class="risk-badge {css}">{icon} Overall Risk: {report.risk_level.upper()}</span>\n\n'
    md += f"### Patient Summary\n{report.patient_summary}\n\n---\n\n"
    md += "### ⚠️ Drug Interaction Warnings\n"
    for w in report.interaction_warnings:
        md += f"- ⚠️ {w}\n"
    md += "\n---\n\n### 📋 Evidence-Based Recommendations\n"
    for r in report.recommendations:
        conf = {"high": "🟢 High", "moderate": "🟡 Moderate", "low": "🔴 Low"}.get(r.confidence.lower(), r.confidence)
        md += f"\n> **{r.recommendation}**\n>\n> *Confidence:* {conf} · *Evidence:* `{', '.join(r.citation_chunk_ids)}`\n"
    md += "\n---\n\n### 🚩 Items Requiring Clinician Attention\n"
    for f in report.flags_for_clinician:
        md += f"- 🚩 **{f}**\n"
    md += f"\n---\n\n*{report.disclaimer}*"
    return md


def run_pipeline_stage(patient_text: str = None, resume_value: str = None):
    """Run pipeline (initial or resume) and return results."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    trace = st.session_state.get("agent_trace", "")
    completed = set(st.session_state.get("completed_steps", []))

    if patient_text:
        # Initial run
        initial_state = {
            "messages": [], "raw_case": patient_text,
            "patient_profile": None, "drug_interactions": [],
            "risk_level": None, "medications_checked": [],
            "normalization_failures": [], "guideline_matches": [],
            "clinical_report": None, "agent_errors": [],
        }
        stream = pipeline.stream(initial_state, config=config, stream_mode="updates")
    else:
        # Resume after HITL
        stream = pipeline.stream(Command(resume=resume_value), config=config, stream_mode="updates")

    for event in stream:
        for node, output in event.items():
            if node == "__interrupt__":
                continue
            if isinstance(output, dict):
                for msg in output.get("messages", []):
                    content = msg.content if hasattr(msg, "content") else msg.get("content", "")
                    if content:
                        trace += f"[{node}] {content}\n\n"
                        if "Successfully extracted" in content:
                            completed.add("intake")
                        elif "Checked" in content and "medication" in content:
                            completed.add("ddi")
                        elif "Searched" in content and "queries" in content:
                            completed.add("rag")
                        elif "Report generated" in content:
                            completed.add("synthesis")

    st.session_state.agent_trace = trace
    st.session_state.completed_steps = list(completed)

    # Check for interrupt
    snapshot = pipeline.get_state(config)
    if snapshot.next:
        interrupted = snapshot.next[0]
        interrupt_msg = ""
        if snapshot.tasks and snapshot.tasks[0].interrupts:
            interrupt_msg = snapshot.tasks[0].interrupts[0].value

        if "ddi" in interrupted.lower():
            st.session_state.phase = "hitl_ddi"
            st.session_state.hitl_type = "ddi"
        else:
            st.session_state.phase = "hitl_synthesis"
            st.session_state.hitl_type = "synthesis"
        st.session_state.interrupt_msg = interrupt_msg
        return

    # Complete
    final = pipeline.get_state(config)
    report = final.values.get("clinical_report")
    st.session_state.final_report = report
    st.session_state.phase = "complete"


# ── Session state init ────────────────────────────────────────────────────────
defaults = {
    "phase": "idle",
    "thread_id": "",
    "completed_steps": [],
    "agent_trace": "",
    "interrupt_msg": "",
    "hitl_type": "",
    "final_report": None,
    "patient_text": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Example cases ─────────────────────────────────────────────────────────────
EXAMPLES = {
    "TC-001: Elderly Diabetic + CKD + AF": (
        "65-year-old male presenting with fatigue and dizziness. Past medical history: "
        "Type 2 Diabetes Mellitus (10 years), Atrial Fibrillation, Hypertension. "
        "Medications: Metformin 1000mg BID, Glipizide 5mg daily, Warfarin 5mg daily. "
        "No known drug allergies. Labs: HbA1c 7.8%, Creatinine 1.8 mg/dL, "
        "eGFR 38 mL/min, INR 3.2, Potassium 4.9 mEq/L. BP: 155/95 mmHg."
    ),
    "TC-002: Clean Case (No DDIs)": (
        "42-year-old female with well-controlled hypertension. Amlodipine 5mg daily. "
        "No other medications. No allergies. BP 128/82, Creatinine 0.9, K 4.1. "
        "Asymptomatic, routine follow-up."
    ),
    "TC-003: Polypharmacy + Brand Names": (
        "71-year-old male with COPD, Type 2 Diabetes, depression, chronic back pain. "
        "Meds: Glucophage 500mg BID, Advil 400mg TID, Zoloft 100mg daily, "
        "Lasix 40mg daily, Lisinopril 20mg daily. Allergic to Penicillin and Sulfa drugs. "
        "Labs: FBS 165, HbA1c 8.1%, Cr 1.4, K 3.4 (low). "
        "Symptoms: persistent cough, shortness of breath, leg swelling."
    ),
}


# =============================================================
# UI LAYOUT
# =============================================================

# ── Hero ──
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent Clinical Decision Support</div>
    <h1>Medi<span>Agent</span></h1>
    <p class="hero-sub">
        Enter patient clinical notes to receive AI-assisted, evidence-based
        guideline recommendations and drug interaction checks.
    </p>
    <div class="hero-badges">
        <span class="hero-badge">⚕️ Healthcare Provider Use</span>
        <span class="hero-badge">🔒 Citation-Grounded</span>
        <span class="hero-badge">👤 Human-in-the-Loop</span>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Main layout ──
col_input, col_spacer, col_pipeline = st.columns([5, 0.5, 4])

with col_input:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    # Example selector — callback writes directly to the text_area's widget key
    def _on_example_change():
        choice = st.session_state.example_select
        if choice != "— Select a test case —":
            st.session_state.patient_input_area = EXAMPLES[choice]

    st.selectbox(
        "LOAD EXAMPLE CASE",
        ["— Select a test case —"] + list(EXAMPLES.keys()),
        key="example_select",
        on_change=_on_example_change,
    )

    patient_text = st.text_area(
        "PATIENT CLINICAL NOTES",
        height=220,
        placeholder=(
            "Enter patient information including:\n"
            "• Demographics (age, sex)\n"
            "• Medical history and conditions\n"
            "• Current medications (name, dosage, frequency)\n"
            "• Known drug allergies\n"
            "• Recent lab results\n"
            "• Current symptoms"
        ),
        key="patient_input_area",
    )

    st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
    run_btn = st.button("⚡ Analyze Patient Case", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)

    steps_done = set(st.session_state.completed_steps)
    phase = st.session_state.phase

    def step_state(name):
        if name in steps_done:
            return "done"
        if phase in ("running_initial", "resuming"):
            ordered = ["intake", "ddi", "rag", "synthesis"]
            for s in ordered:
                if s not in steps_done:
                    return "running" if s == name else "waiting"
        return "waiting"

    step_card("01", "Intake Agent", step_state("intake"), "Extracts structured patient data from clinical notes")
    step_card("02", "DDI Agent", step_state("ddi"), "Checks drug interactions via RxNorm + DDInter 2.0")
    step_card("03", "Guidelines RAG", step_state("rag"), "Retrieves evidence-based clinical guidelines")
    step_card("04", "Synthesis Agent", step_state("synthesis"), "Generates grounded clinical report with citations")


# ── Button handler ──
if run_btn:
    if not patient_text.strip():
        st.warning("Please enter patient clinical notes first.")
    else:
        st.session_state.patient_text = patient_text
        st.session_state.phase = "running_initial"
        st.session_state.thread_id = f"st-{int(time.time())}"
        st.session_state.completed_steps = []
        st.session_state.agent_trace = ""
        st.session_state.final_report = None
        st.rerun()


# ── Pipeline execution ──
if st.session_state.phase == "running_initial":
    with st.spinner("🧠 Analyzing patient case — agents are working..."):
        run_pipeline_stage(patient_text=st.session_state.patient_text)
    st.rerun()

if st.session_state.phase == "resuming":
    resume_val = st.session_state.get("resume_value", "approve")
    with st.spinner("🧠 Resuming pipeline..."):
        run_pipeline_stage(resume_value=resume_val)
    st.rerun()


# ── HITL Review ──
if st.session_state.phase in ("hitl_ddi", "hitl_synthesis"):
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    hitl_type = st.session_state.hitl_type
    if hitl_type == "ddi":
        label = "⚠️ DRUG INTERACTION REVIEW REQUIRED"
    else:
        label = "📋 PRE-SYNTHESIS EVIDENCE REVIEW"

    st.markdown(f"""
    <div class="hitl-card">
        <div class="hitl-label">{label}</div>
        <div class="hitl-content">{st.session_state.interrupt_msg}</div>
    </div>
    """, unsafe_allow_html=True)

    feedback = st.text_input(
        "CLINICAL OVERRIDE / FEEDBACK (OPTIONAL)",
        placeholder="Leave blank to approve, or provide specific clinical feedback...",
        key="hitl_feedback_input",
    )

    col_approve, col_reject = st.columns([2, 1])
    with col_approve:
        st.markdown('<div class="approve-btn">', unsafe_allow_html=True)
        if st.button("✅ Approve & Continue", use_container_width=True, key="approve_btn"):
            st.session_state.phase = "resuming"
            st.session_state.resume_value = "approve"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_reject:
        st.markdown('<div class="reject-btn">', unsafe_allow_html=True)
        if st.button("↩️ Send Feedback", use_container_width=True, key="reject_btn"):
            st.session_state.phase = "resuming"
            st.session_state.resume_value = feedback.strip() or "Clinician requested re-evaluation."
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ── Final Report ──
if st.session_state.phase == "complete" and st.session_state.final_report:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Clinical Report</div>', unsafe_allow_html=True)

    report = st.session_state.final_report
    st.markdown(f"""
    <div class="report-panel">
        <h2>📋 Clinical Decision Support Report</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(format_report(report), unsafe_allow_html=True)

    # Download button
    report_text = format_report(report)
    st.download_button(
        label="⬇ Download Report (.md)",
        data=report_text,
        file_name=f"mediagent_report_{int(time.time())}.md",
        mime="text/markdown",
    )


# ── Agent Trace (collapsed) ──
if st.session_state.agent_trace:
    with st.expander("🔧 Developer Diagnostics — Agent Trace"):
        st.text(st.session_state.agent_trace)


# ── Footer ──
st.markdown("""
<div class="notice">
    MediAgent v1.0 · LangGraph · Gemini 2.5 Flash · DDInter 2.0 · RxNorm · ChromaDB<br>
    ⚕️ For decision support only — all outputs must be verified by a qualified healthcare professional
</div>
""", unsafe_allow_html=True)
