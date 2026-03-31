"""AIC Streamlit Web Interface."""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="AI Consulting System",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource(show_spinner="Initialising consulting engine…")
def get_service():
    from core.consulting_service import ConsultingService
    db_path = str(Path(__file__).parent.parent / "data" / "aic.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path("memory").mkdir(exist_ok=True)
    return ConsultingService(memory_dir="memory", db_path=db_path)


with st.sidebar:
    st.title("🤖 AIC")
    st.markdown("**AI Consulting System**")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Consulting Analysis", "Session History", "System Info"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Phase 5 — Production Build")


# ===========================================================================
# Dashboard
# ===========================================================================
if page == "Dashboard":
    st.title("Dashboard")
    st.markdown("Welcome to the AI Consulting System.")

    svc = get_service()
    sessions = svc.list_sessions(limit=100)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Sessions", len(sessions))
    with c2:
        valid = sum(1 for s in sessions if s.get("validation_status") == "valid")
        st.metric("Validation Pass Rate", f"{valid/len(sessions):.0%}" if sessions else "—")
    with c3:
        avg_conf = sum(s.get("confidence", 0) for s in sessions) / len(sessions) if sessions else 0
        st.metric("Avg Confidence", f"{avg_conf:.0%}" if sessions else "—")

    if sessions:
        st.subheader("Recent Sessions")
        for s in sessions[:5]:
            with st.expander(f"🔹 {s.get('problem','')[:80]}  ({s.get('session_id','')[:8]})"):
                st.write(f"**Validation:** {s.get('validation_status','—')} | "
                         f"**Confidence:** {s.get('confidence',0):.0%}")


# ===========================================================================
# Consulting Analysis
# ===========================================================================
elif page == "Consulting Analysis":
    st.title("Consulting Analysis")
    st.markdown("Describe your business problem to receive a structured consulting report.")

    with st.sidebar:
        st.subheader("Business Context")
        industry = st.selectbox("Industry", [
            "general", "retail", "SaaS", "manufacturing", "healthcare",
            "finance", "logistics", "hospitality", "education",
        ])
        company_size = st.selectbox("Company Size",
                                    ["micro", "sme", "mid-market", "enterprise"], index=1)
        market_position = st.selectbox("Market Position",
                                       ["leader", "challenger", "follower", "niche"], index=1)
        strategic_priorities = st.text_input("Strategic Priorities (comma-separated)",
                                             placeholder="growth, efficiency, innovation")

    problem = st.text_area("Business Problem",
                           placeholder="e.g. We are losing market share…", height=130)

    if st.button("Analyse", type="primary"):
        if not problem.strip():
            st.error("Please enter a business problem.")
        else:
            context = {
                "industry": industry,
                "company_size": company_size,
                "market_position": market_position,
                "strategic_priorities": [p.strip() for p in strategic_priorities.split(",") if p.strip()],
            }
            with st.spinner("Running autonomous analysis…"):
                svc = get_service()
                try:
                    report = svc.analyze(problem.strip(), context)
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    st.stop()

            st.success(f"Analysis complete — Session `{report.session_id[:8]}`")

            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{report.confidence:.0%}")
            m2.metric("Validation", report.validation_status.capitalize())
            m3.metric("Past Sessions Used", report.past_sessions_used)

            with st.expander("🧠 Autonomous Reasoning", expanded=True):
                st.markdown(report.reasoning_summary)
                st.caption(f"Validation: **{report.validation_status}**")

            with st.expander("🔍 Self-Critique"):
                st.markdown(report.critique_summary)
                st.caption(f"Critique score: {report.critique_score:.2f}")

            if report.recommended_frameworks:
                with st.expander("📋 Recommended Frameworks"):
                    for fw in report.recommended_frameworks:
                        st.markdown(f"- **{fw.replace('_',' ').title()}**")

            with st.expander("📊 Framework Analyses", expanded=True):
                for fw_name, analysis in report.framework_analyses.items():
                    st.subheader(fw_name.replace("_", " ").title())
                    if isinstance(analysis, dict):
                        for k, v in analysis.items():
                            if k not in ("framework", "problem"):
                                label = k.replace("_", " ").title()
                                if isinstance(v, list):
                                    st.markdown(f"**{label}:**")
                                    for item in v:
                                        st.markdown(f"  - {item}")
                                elif isinstance(v, dict):
                                    st.markdown(f"**{label}:**")
                                    st.json(v)
                                else:
                                    st.markdown(f"**{label}:** {v}")
                    st.divider()


# ===========================================================================
# Session History
# ===========================================================================
elif page == "Session History":
    st.title("Session History")
    svc = get_service()
    sessions = svc.list_sessions(limit=50)

    if not sessions:
        st.info("No sessions yet. Run a Consulting Analysis to get started.")
    else:
        for s in sessions:
            with st.expander(f"[{s.get('session_id','')[:8]}] {s.get('problem','')[:70]}"):
                st.write(f"**Confidence:** {s.get('confidence',0):.0%}  |  "
                         f"**Validation:** {s.get('validation_status','—')}  |  "
                         f"**Created:** {s.get('created_at','—')}")
                if s.get("reasoning_summary"):
                    st.markdown("**Reasoning:** " + s["reasoning_summary"])


# ===========================================================================
# System Info
# ===========================================================================
elif page == "System Info":
    st.title("System Info")
    st.markdown("""
### AIC — AI Consulting System

**Architecture:**
- Autonomous pipeline: GoalPlanner → IntentionManager → ThoughtProcessor → ReasoningValidator → SelfCritic
- Consulting frameworks: SWOT, Porter's Five Forces, McKinsey 7S, MECE, Ansoff, BCG
- Persistent storage: SQLite
- REST API: FastAPI on port 8000

**Phase 5 Features:**
- PyTorch dead code removed — pure Python consulting engine
- End-to-end consulting analysis via CLI, Streamlit, and REST API
- Self-learning: past sessions inform new analyses
- Docker-ready deployment
    """)
    try:
        from core.engine import create_aic_system
        engine = create_aic_system()
        st.success(f"✓ Engine: {type(engine).__name__}")
    except Exception as exc:
        st.error(f"❌ Engine error: {exc}")
