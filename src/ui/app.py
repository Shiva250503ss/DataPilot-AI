"""
DataPilot AI Pro - Streamlit UI
================================
Main user interface for the data science platform.
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="DataPilot AI Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Configuration
API_URL = "http://localhost:8000"


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🚀 DataPilot AI Pro")
    st.markdown("### AI-Powered Autonomous Data Science Platform")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.shields.io/badge/Version-1.0.0-blue")
        st.markdown("---")
        
        mode = st.radio(
            "Analysis Mode",
            ["Chat Mode (Autonomous)", "Guided Mode (Interactive)"],
            help="Chat mode runs automatically. Guided mode asks for approval."
        )
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        if "task_id" in st.session_state:
            st.success(f"Active Task: {st.session_state['task_id'][:8]}...")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 Upload", "📈 Results", "💬 Chat"])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_results_tab()
    
    with tab3:
        render_chat_tab()


def render_upload_tab():
    """Render the file upload section."""
    st.header("Upload Your Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Drag and drop your CSV file",
            type=["csv"],
            help="Upload a CSV file to begin analysis",
        )
        
        if uploaded_file is not None:
            # Preview data
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
            
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column info
            with st.expander("📋 Column Information"):
                col_info = pd.DataFrame({
                    "Type": df.dtypes.astype(str),
                    "Non-Null": df.notna().sum(),
                    "Null %": (df.isna().sum() / len(df) * 100).round(2),
                    "Unique": df.nunique(),
                })
                st.dataframe(col_info, use_container_width=True)
            
            # Natural language prompt
            prompt = st.text_area(
                "Optional: Describe what you want to analyze",
                placeholder="e.g., 'Focus on predicting customer churn' or 'Find patterns in sales data'",
                height=100,
            )
            
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                with st.spinner("Uploading and analyzing..."):
                    start_analysis(uploaded_file, prompt)
    
    with col2:
        st.markdown("### How it works")
        st.markdown("""
        1. **Upload** your CSV file
        2. **Describe** your analysis goal (optional)
        3. **Click** Start Analysis
        4. **View** automated insights
        
        ---
        
        **Supported:**
        - CSV files (UTF-8)
        - Up to 1M rows
        - Classification & Regression
        """)


def render_results_tab():
    """Render the results section."""
    st.header("Analysis Results")
    
    if "task_id" not in st.session_state:
        st.info("Upload a dataset to see results here.")
        return
    
    # Fetch results
    task_id = st.session_state["task_id"]
    
    # Status check
    status_response = requests.get(f"{API_URL}/status/{task_id}")
    
    if status_response.status_code != 200:
        st.error("Could not fetch status")
        return
    
    status = status_response.json()
    
    # Progress bar
    if status["status"] == "analyzing":
        st.progress(status.get("progress", 0))
        st.info(f"Current stage: {status.get('current_stage', 'Processing...')}")
        st.button("🔄 Refresh", on_click=lambda: None)
        return
    
    if status["status"] == "error":
        st.error(f"Analysis failed: {status.get('error')}")
        return
    
    if status["status"] == "completed":
        st.success("✅ Analysis Complete!")
        
        # Fetch full results
        results_response = requests.get(f"{API_URL}/results/{task_id}")
        
        if results_response.status_code == 200:
            results = results_response.json()
            
            # Summary cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rows", f"{results['summary'].get('rows', 0):,}")
            with col2:
                st.metric("Columns", results['summary'].get('columns', 0))
            with col3:
                st.metric("Best Model", results['summary'].get('best_model', 'N/A'))
            with col4:
                best_score = max(
                    [m.get('f1_score', 0) for m in results['metrics'].values()],
                    default=0
                )
                st.metric("F1 Score", f"{best_score:.3f}")
            
            st.markdown("---")
            
            # Model comparison
            st.subheader("📊 Model Comparison")
            
            if results['metrics']:
                metrics_df = pd.DataFrame(results['metrics']).T
                metrics_df.index.name = "Model"
                st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True)
            
            # Feature importance
            st.subheader("🔍 Feature Importance")
            
            if results['feature_importance']:
                importance_df = pd.DataFrame.from_dict(
                    results['feature_importance'],
                    orient='index',
                    columns=['importance']
                ).sort_values('importance', ascending=True).tail(15)
                
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y=importance_df.index,
                    orientation='h',
                    title='Top 15 Important Features',
                    color='importance',
                    color_continuous_scale='Viridis',
                )
                st.plotly_chart(fig, use_container_width=True)


def render_chat_tab():
    """Render the chat interface."""
    st.header("💬 Ask Questions")
    
    if "task_id" not in st.session_state:
        st.info("Complete an analysis first to ask questions about results.")
        return
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your analysis..."):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response (mock for now)
        with st.chat_message("assistant"):
            response = f"Based on your analysis: {prompt}"
            st.write(response)
        
        st.session_state["messages"].append({"role": "assistant", "content": response})


def start_analysis(uploaded_file, prompt: Optional[str] = None):
    """Start the analysis pipeline."""
    try:
        # Upload file
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        upload_response = requests.post(f"{API_URL}/upload", files=files)
        
        if upload_response.status_code != 200:
            st.error("Upload failed")
            return
        
        upload_data = upload_response.json()
        task_id = upload_data["task_id"]
        
        # Start analysis
        analyze_response = requests.post(
            f"{API_URL}/analyze",
            json={
                "task_id": task_id,
                "mode": "chat",
                "prompt": prompt,
            }
        )
        
        if analyze_response.status_code != 200:
            st.error("Analysis failed to start")
            return
        
        st.session_state["task_id"] = task_id
        st.success(f"Analysis started! Task ID: {task_id[:8]}...")
        st.info("Switch to the Results tab to view progress.")
        
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API. Make sure the backend is running.")
    except Exception as e:
        st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
