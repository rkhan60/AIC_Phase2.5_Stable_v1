"""AIC Streamlit application — 4-page consulting interface."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on path when launched directly
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

st.set_page_config(
    page_title="AIC — AI Consulting System",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading consulting engine...")
def _get_service():
    from core.consulting_service import ConsultingService
    base = _ROOT
    return ConsultingService(
        memory_dir=str(base / "memory"),
        db_path=str(base / "data" / "aic.db"),
    )


# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
with st.sidebar:
    st.title("AIC")
    st.caption("AI Consulting System v5.0")
    page = st.radio(
        "Navigate",
        ["Dashboard", "Consulting Analysis", "Session History", "System Info"],
    )
    st.divider()
    st.caption("Persistent sessions stored in SQLite.")


# ==================================================================
# Page: Dashboard
# ==================================================================
if page == "Dashboard":
    st.header("Dashboard")

    svc = _get_service()
    sessions = svc.list_sessions(limit=100)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sessions", len(sessions))
    validated = sum(1 for s in sessions if s.get("validation_status") == "valid")
    col2.metric("Validated Analyses", validated)
    avg_conf = (
        round(sum(s.get("confidence", 0) for s in sessions) / len(sessions), 2)
        if sessions else 0.0
    )
    col3.metric("Avg Confidence", f"{avg_conf:.0%}")

    if sessions:
        st.subheader("Recent Sessions")
        import pandas as pd
        df = pd.DataFrame([
            {
                "Session ID": s["session_id"][:8] + "...",
                "Problem": s.get("problem", "")[:60] + "...",
                "Confidence": f"{s.get('confidence', 0):.0%}",
                "Validation": s.get("validation_status", ""),
                "Date": s.get("created_at", "")[:19],
            }
            for s in sessions[:10]
        ])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No sessions yet. Run a consulting analysis to get started.")


# ==================================================================
# Page: Consulting Analysis
# ==================================================================
elif page == "Consulting Analysis":
    st.header("Consulting Analysis")

    with st.sidebar:
        st.subheader("Business Context")
        industry = st.text_input("Industry", value="general")
        company_size = st.selectbox(
            "Company Size",
            ["micro", "sme", "mid-market", "enterprise"],
            index=1,
        )
        market_position = st.selectbox(
            "Market Position",
            ["leader", "challenger", "follower", "niche"],
            index=1,
        )
        priorities = st.text_area(
            "Strategic Priorities (one per line)", height=80
        )

    problem = st.text_area(
        "Describe your business problem",
        height=150,
        placeholder="e.g. We are losing market share to low-cost competitors in the mid-market segment...",
    )

    if st.button("Analyze", type="primary", disabled=not problem.strip()):
        context = {
            "industry": industry,
            "company_size": company_size,
            "market_position": market_position,
            "strategic_priorities": [p.strip() for p in priorities.splitlines() if p.strip()],
        }
        with st.spinner("Running analysis..."):
            svc = _get_service()
            try:
                report = svc.analyze(problem, context)
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.stop()

        col1, col2, col3 = st.columns(3)
        col1.metric("Confidence", f"{report.confidence:.0%}")
        col2.metric("Validation", report.validation_status)
        col3.metric("Session", report.session_id[:8] + "...")
        if report.past_sessions_used:
            st.caption(f"Informed by {report.past_sessions_used} similar past session(s).")

        tab1, tab2, tab3 = st.tabs(["Reasoning", "Critique", "Framework Analyses"])

        with tab1:
            st.markdown(report.reasoning_summary)

        with tab2:
            st.markdown(report.critique_summary)
            if report.recommended_frameworks:
                st.subheader("Recommended Frameworks")
                for fw in report.recommended_frameworks:
                    st.markdown(f"- `{fw}`")

        with tab3:
            for fw_name, analysis in report.framework_analyses.items():
                if fw_name in ("problem", "context_summary", "aggregated_recommendations"):
                    continue
                with st.expander(fw_name.upper().replace("_", " ")):
                    if isinstance(analysis, dict):
                        for k, v in analysis.items():
                            if k not in ("framework", "problem"):
                                st.write(f"**{k}:** {v}")
                    else:
                        st.write(analysis)

            agg = report.framework_analyses.get("aggregated_recommendations", [])
            if agg:
                st.subheader("Aggregated Recommendations")
                for rec in agg:
                    st.markdown(f"- {rec}")


# ==================================================================
# Page: Session History
# ==================================================================
elif page == "Session History":
    st.header("Session History")
    svc = _get_service()
    sessions = svc.list_sessions(limit=50)

    if not sessions:
        st.info("No sessions recorded yet.")
    else:
        for s in sessions:
            label = f"{s.get('created_at', '')[:19]} | {s.get('problem', '')[:60]}"
            with st.expander(label):
                st.write(f"**Session ID:** {s['session_id']}")
                st.write(f"**Confidence:** {s.get('confidence', 0):.0%}")
                st.write(f"**Validation:** {s.get('validation_status', 'N/A')}")
                st.write(f"**Reasoning:**\n{s.get('reasoning_summary', '')}")
                st.write(f"**Critique:**\n{s.get('critique_summary', '')}")


# ==================================================================
# Page: System Info
# ==================================================================
elif page == "System Info":
    st.header("System Info")
    st.markdown("""
| Component | Description |
|-----------|-------------|
| **ConsultingService** | Central facade composing pipeline + framework engine |
| **AutonomousPipeline** | Reasoning, validation, self-critique loop |
| **ConsultingFrameworkEngine** | SWOT, Porter's 5 Forces, McKinsey 7S, MECE, Ansoff, BCG |
| **SessionRepository** | SQLite persistence for all analysis sessions |
| **REST API** | FastAPI — `POST /api/v1/analyze`, session CRUD, health check |
| **Streamlit** | This interface |
""")
    st.subheader("Start REST API")
    st.code("uvicorn api.main:app --reload --port 8000", language="bash")
    st.subheader("Docker Compose")
    st.code("docker-compose up --build", language="bash")
