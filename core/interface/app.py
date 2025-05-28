import streamlit as st
import torch
import pandas as pd
import plotly.graph_objects as go
from logic_engine import create_aic_system, ConsultingRole, ReasoningType

# Page config
st.set_page_config(
    page_title="AI Consulting System",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("🧠 AI Consulting System - Reasoning-Based Memory")
st.markdown("""
This advanced AI system provides strategic business consulting using sophisticated 
reasoning pathways and human-like memory integration.
""")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.training, _ = create_aic_system()
    st.session_state.history = []
    st.session_state.memory_stats = {}

# Sidebar
with st.sidebar:
    st.header("Reasoning Configuration")
    
    # Reasoning pathway selection
    reasoning_path = st.selectbox(
        "Primary Reasoning Pathway",
        ["Deductive", "Inductive", "Abductive", "Analogical", "Causal", "Counterfactual"]
    )
    
    # Memory integration settings
    st.subheader("Memory Integration")
    use_episodic = st.checkbox("Use Case History", value=True)
    use_semantic = st.checkbox("Use Business Frameworks", value=True)
    use_reasoning = st.checkbox("Use Reasoning Patterns", value=True)
    
    # Industry and role selection
    industry = st.selectbox(
        "Industry Context",
        ["Technology", "Healthcare", "Financial Services", "Manufacturing", "Retail", "Energy"]
    )
    
    role = st.selectbox(
        "Consulting Role",
        [role.value for role in ConsultingRole]
    )

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Problem Description")
    problem_description = st.text_area(
        "Describe your business problem",
        height=150,
        placeholder="Enter a detailed description of your business challenge..."
    )
    
    # Context parameters
    st.subheader("Problem Context")
    complexity = st.select_slider(
        "Problem Complexity",
        options=["Simple", "Moderate", "Complex", "Very Complex"]
    )
    
    uncertainty = st.select_slider(
        "Uncertainty Level",
        options=["Low", "Medium", "High", "Very High"]
    )
    
    time_horizon = st.select_slider(
        "Time Horizon",
        options=["Immediate", "Short-term", "Medium-term", "Long-term"]
    )

with col2:
    st.subheader("Memory System Status")
    if st.session_state.memory_stats:
        stats = st.session_state.memory_stats
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Episodic Memory", f"{stats['episodic_usage']:.2f}")
            st.metric("Semantic Memory", f"{stats['semantic_usage']:.2f}")
        with col_b:
            st.metric("Reasoning Memory", f"{stats['reasoning_usage']:.2f}")
            st.metric("Working Memory", f"{stats['working_memory_load']:.2f}")

# Analysis button
if st.button("Generate Analysis", type="primary"):
    if problem_description:
        with st.spinner("Analyzing with reasoning-based memory..."):
            try:
                # Create input tensor (simplified)
                input_tensor = torch.randint(0, 50000, (1, 512))
                
                # Get analysis from model
                analysis = st.session_state.model(
                    input_tensor,
                    attention_mask=None,
                    problem_context={
                        'industry': industry,
                        'complexity': complexity,
                        'uncertainty': uncertainty,
                        'time_horizon': time_horizon,
                        'reasoning_path': reasoning_path.lower(),
                        'memory_config': {
                            'use_episodic': use_episodic,
                            'use_semantic': use_semantic,
                            'use_reasoning': use_reasoning
                        }
                    }
                )
                
                # Update memory stats
                st.session_state.memory_stats = st.session_state.model.memory_system.get_memory_stats()
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Reasoning Process", 
                    "Analysis & Recommendations",
                    "Memory Integration",
                    "Confidence & Metrics"
                ])
                
                with tab1:
                    st.markdown("### Reasoning Pathway Analysis")
                    # Display reasoning weights
                    reasoning_explanations = st.session_state.model.reasoning_engine.explain_reasoning(
                        analysis['reasoning_weights']
                    )
                    for explanation in reasoning_explanations:
                        st.info(explanation)
                    
                    # Reasoning flow visualization
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[path for path in ["Deductive", "Inductive", "Abductive", "Analogical", "Causal", "Counterfactual"]],
                            y=analysis['reasoning_weights'][0].tolist(),
                            name="Reasoning Weights"
                        )
                    ])
                    fig.update_layout(title="Reasoning Pathway Utilization")
                    st.plotly_chart(fig)
                
                with tab2:
                    st.markdown("### Strategic Analysis")
                    st.info(str(analysis['analysis']))
                    
                    # Implementation roadmap
                    timeline_df = pd.DataFrame({
                        'Phase': ['Analysis', 'Strategy', 'Implementation', 'Review'],
                        'Start': [1, 2, 4, 7],
                        'Duration': [1, 2, 3, 1]
                    })
                    fig = go.Figure(go.Bar(
                        x=timeline_df.Duration,
                        y=timeline_df.Phase,
                        orientation='h'
                    ))
                    fig.update_layout(title='Implementation Timeline')
                    st.plotly_chart(fig)
                
                with tab3:
                    st.markdown("### Memory System Integration")
                    col_mem1, col_mem2 = st.columns(2)
                    
                    with col_mem1:
                        st.metric("Memory Retrieval Confidence", f"{analysis['memory_confidence']:.2%}")
                        st.metric("Pattern Match Score", f"{analysis['pattern_match']:.2%}")
                    
                    with col_mem2:
                        st.metric("Knowledge Integration", f"{analysis['knowledge_score']:.2%}")
                        st.metric("Learning Rate", f"{analysis['learning_rate']:.2%}")
                
                with tab4:
                    st.markdown("### Analysis Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': ['Solution Confidence', 'Implementation Complexity', 'Resource Requirements', 'Risk Level'],
                        'Score': [0.85, 0.65, 0.45, 0.35]
                    })
                    st.dataframe(metrics_df)
                    
                    # Confidence visualization
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = analysis['confidence'],
                        title = {'text': "Solution Confidence"},
                        gauge = {'axis': {'range': [0, 1]}}
                    ))
                    st.plotly_chart(fig)
                
                # Add to history
                st.session_state.history.append({
                    'timestamp': pd.Timestamp.now(),
                    'problem': problem_description,
                    'analysis': analysis,
                    'reasoning_path': reasoning_path,
                    'memory_stats': st.session_state.memory_stats.copy()
                })
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
    else:
        st.warning("Please provide a problem description.")

# History section with memory evolution
if st.session_state.history:
    st.subheader("Analysis History & Memory Evolution")
    for i, item in enumerate(reversed(st.session_state.history[-5:])):
        with st.expander(f"Analysis {len(st.session_state.history)-i}: {item['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
            col_hist1, col_hist2 = st.columns(2)
            
            with col_hist1:
                st.markdown("**Problem Description:**")
                st.text(item['problem'][:100] + "...")
                st.markdown("**Primary Reasoning:**")
                st.info(item['reasoning_path'])
            
            with col_hist2:
                st.markdown("**Memory State:**")
                mem_stats = item['memory_stats']
                st.metric("Episodic", f"{mem_stats['episodic_usage']:.2f}")
                st.metric("Semantic", f"{mem_stats['semantic_usage']:.2f}")
                st.metric("Reasoning", f"{mem_stats['reasoning_usage']:.2f}") 