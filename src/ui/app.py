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
    tab1, tab2, tab3 = st.tabs(["Data Source", "Results", "Chat"])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_results_tab()
    
    with tab3:
        render_chat_tab()


def render_upload_tab():
    """Render the data source selection and upload section."""
    st.header("Connect Your Data")

    source_type = st.radio(
        "Data Source",
        ["File Upload (CSV / Excel)", "Database Connection"],
        horizontal=True,
    )

    if source_type == "File Upload (CSV / Excel)":
        _render_file_upload()
    else:
        _render_db_connect()


def _render_file_upload():
    """CSV and Excel file upload."""
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Drag and drop your CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Upload a CSV or Excel file to begin analysis",
        )

        if uploaded_file is not None:
            # Preview data
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {uploaded_file.name}")
            st.dataframe(df.head(10), use_container_width=True)

            with st.expander("Column Information"):
                col_info = pd.DataFrame({
                    "Type": df.dtypes.astype(str),
                    "Non-Null": df.notna().sum(),
                    "Null %": (df.isna().sum() / len(df) * 100).round(2),
                    "Unique": df.nunique(),
                })
                st.dataframe(col_info, use_container_width=True)

            prompt = st.text_area(
                "Optional: Describe what you want to analyze",
                placeholder="e.g., 'Focus on predicting customer churn'",
                height=100,
            )

            if st.button("Start Analysis", type="primary", use_container_width=True):
                with st.spinner("Uploading and starting analysis..."):
                    start_analysis(uploaded_file, prompt)

    with col2:
        st.markdown("### How it works")
        st.markdown("""
        1. **Upload** your CSV or Excel file
        2. **Describe** your analysis goal (optional)
        3. **Click** Start Analysis
        4. **View** automated insights

        ---

        **Supported Formats:**
        - CSV (UTF-8, comma-separated)
        - Excel (.xlsx, .xls)
        - Database tables (PostgreSQL, MySQL, SQLite)
        - Up to 1M+ rows
        - Classification & Regression
        """)


def _render_db_connect():
    """Database connection UI."""
    st.subheader("Connect to a Database")

    col1, col2 = st.columns([2, 1])

    with col1:
        db_type = st.selectbox(
            "Database Type",
            ["PostgreSQL", "MySQL", "SQLite", "Other (custom URL)"],
        )

        if db_type == "PostgreSQL":
            host = st.text_input("Host", value="localhost")
            port = st.number_input("Port", value=5432)
            dbname = st.text_input("Database Name")
            user = st.text_input("Username")
            password = st.text_input("Password", type="password")
            connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        elif db_type == "MySQL":
            host = st.text_input("Host", value="localhost")
            port = st.number_input("Port", value=3306)
            dbname = st.text_input("Database Name")
            user = st.text_input("Username")
            password = st.text_input("Password", type="password")
            connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"
        elif db_type == "SQLite":
            db_path = st.text_input("Database File Path", placeholder="/path/to/database.db")
            connection_string = f"sqlite:///{db_path}"
        else:
            connection_string = st.text_input(
                "SQLAlchemy Connection URL",
                placeholder="dialect+driver://user:pass@host/db",
            )

        st.text_input("Connection URL (preview)", value=connection_string, disabled=True)

        if st.button("Connect & List Tables", use_container_width=True):
            try:
                response = requests.post(
                    f"{API_URL}/connect-db",
                    json={"connection_string": connection_string},
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Connected! Found {len(data['tables'])} tables.")
                    st.session_state["db_tables"] = data["tables"]
                    st.session_state["db_conn_string"] = connection_string
                else:
                    st.error(f"Connection failed: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Error: {e}")

        # Table selection after connection
        if "db_tables" in st.session_state:
            table = st.selectbox("Select Table to Analyze", st.session_state["db_tables"])
            prompt = st.text_area(
                "Optional: Describe what you want to analyze",
                placeholder="e.g., 'Find top revenue customers'",
                height=80,
            )

            if st.button("Load Table & Start Analysis", type="primary", use_container_width=True):
                with st.spinner("Loading table from database..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/connect-db",
                            json={
                                "connection_string": st.session_state["db_conn_string"],
                                "table": table,
                            },
                        )
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state["task_id"] = data["task_id"]
                            st.success(f"Loaded {data['rows']:,} rows from '{table}'")
                            st.info("Switch to the Results tab to monitor progress.")

                            # Trigger analysis
                            requests.post(f"{API_URL}/analyze", json={
                                "task_id": data["task_id"],
                                "mode": "chat",
                                "prompt": prompt,
                            })
                        else:
                            st.error(f"Failed: {response.json().get('detail')}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        # NL-to-SQL section
        st.markdown("---")
        st.subheader("Natural Language SQL Query")
        st.markdown("Ask a question about your database in plain English.")

        nl_question = st.text_input(
            "Your question",
            placeholder="e.g., 'Show me the top 10 customers by revenue last month'",
        )

        if st.button("Run NL-to-SQL Query", use_container_width=True):
            if not nl_question:
                st.warning("Please enter a question.")
            elif "db_conn_string" not in st.session_state:
                st.warning("Connect to a database first.")
            else:
                with st.spinner("Generating and executing SQL..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/nl-sql",
                            json={
                                "connection_string": st.session_state["db_conn_string"],
                                "question": nl_question,
                            },
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.success("Query executed successfully!")
                            st.code(result["sql"], language="sql")
                            st.caption(f"Explanation: {result['explanation']}")
                            if result["results"]:
                                st.dataframe(pd.DataFrame(result["results"]), use_container_width=True)
                                st.caption(f"{result['row_count']} rows returned")
                        else:
                            st.error(f"Query failed: {response.json().get('detail')}")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with col2:
        st.markdown("### Supported Databases")
        st.markdown("""
        - **PostgreSQL** — enterprise default
        - **MySQL / MariaDB** — web applications
        - **SQLite** — local / embedded
        - Any **SQLAlchemy**-compatible DB

        ---

        ### NL-to-SQL Examples
        - *"Show sales by region last quarter"*
        - *"Top 5 customers by order count"*
        - *"Find users who haven't logged in for 30 days"*
        - *"Monthly revenue trend for 2024"*
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
