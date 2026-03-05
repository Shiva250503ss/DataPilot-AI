"""
DataPilot AI Pro - FastAPI Backend
===================================
REST API for the data science platform.

Endpoints:
- POST /upload        - Upload CSV or Excel dataset for analysis
- POST /connect-db    - Connect to a database via connection string
- POST /analyze       - Start analysis pipeline
- GET  /status/{id}   - Check task status
- GET  /results/{id}  - Get analysis results
- POST /predict       - Make predictions with trained model
- POST /nl-sql        - Natural language to SQL query
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import io
import uuid
from loguru import logger

# Create FastAPI app
app = FastAPI(
    title="DataPilot AI Pro",
    description="AI-Powered Autonomous Data Science Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory task storage (use Redis in production)
tasks: Dict[str, Dict[str, Any]] = {}


# Pydantic models
class AnalysisRequest(BaseModel):
    """Request model for analysis."""
    task_id: str
    mode: str = "chat"
    prompt: Optional[str] = None


class DatabaseConnectRequest(BaseModel):
    """Request model for database connection."""
    connection_string: str
    table: Optional[str] = None  # If None, lists available tables


class NLSQLRequest(BaseModel):
    """Request model for natural language to SQL."""
    connection_string: str
    question: str


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    task_id: str
    data: List[Dict[str, Any]]


class TaskStatus(BaseModel):
    """Response model for task status."""
    task_id: str
    status: str
    progress: float
    current_stage: Optional[str] = None
    error: Optional[str] = None


class AnalysisResult(BaseModel):
    """Response model for analysis results."""
    task_id: str
    status: str
    summary: Dict[str, Any]
    metrics: Dict[str, Any]
    feature_importance: Dict[str, float]
    visualizations: List[str]


# API Routes
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "DataPilot AI Pro",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a dataset for analysis.

    Accepts CSV and Excel (.xlsx, .xls) files and returns a task_id for tracking.
    """
    filename = file.filename or ""
    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")

    try:
        content = await file.read()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        else:
            # Excel support via openpyxl
            df = pd.read_excel(io.BytesIO(content))

        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": "uploaded",
            "progress": 0.0,
            "data": df,
            "filename": filename,
            "source": "file",
            "rows": len(df),
            "columns": len(df.columns),
        }

        logger.info(f"File uploaded: {filename} ({len(df)} rows)")

        return {
            "task_id": task_id,
            "filename": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
        }

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/connect-db")
async def connect_database(request: DatabaseConnectRequest):
    """
    Connect to a database and load a table as a DataFrame for analysis.

    Supports any SQLAlchemy-compatible database:
      - PostgreSQL: postgresql://user:pass@host/db
      - MySQL:      mysql+pymysql://user:pass@host/db
      - SQLite:     sqlite:///path/to/file.db

    If no table is specified, returns the list of available tables.
    """
    try:
        from sqlalchemy import create_engine, inspect, text

        engine = create_engine(request.connection_string)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not request.table:
            return {"status": "connected", "tables": tables}

        if request.table not in tables:
            raise HTTPException(status_code=400, detail=f"Table '{request.table}' not found. Available: {tables}")

        df = pd.read_sql_table(request.table, engine)

        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": "uploaded",
            "progress": 0.0,
            "data": df,
            "filename": f"{request.table} (database)",
            "source": "database",
            "connection_string": request.connection_string,
            "rows": len(df),
            "columns": len(df.columns),
        }

        logger.info(f"DB table loaded: {request.table} ({len(df)} rows)")

        return {
            "task_id": task_id,
            "table": request.table,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DB connection error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/nl-sql")
async def natural_language_to_sql(request: NLSQLRequest):
    """
    Convert a natural language question to SQL and execute it.

    Uses GPT-4 with dynamic schema injection and a self-correction
    loop to generate accurate, executable SQL for any connected database.
    """
    try:
        from ..agents.nl_sql_agent import NLSQLAgent

        agent = NLSQLAgent(connection_string=request.connection_string)
        result = await agent.query(request.question)

        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])

        results_data = None
        if result["results"] is not None:
            results_data = result["results"].to_dict(orient="records")

        return {
            "question": request.question,
            "sql": result["sql"],
            "explanation": result["explanation"],
            "results": results_data,
            "row_count": result.get("row_count", 0),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NL-SQL error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze")
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start analysis pipeline for uploaded dataset.
    
    Returns immediately and runs analysis in background.
    """
    if request.task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[request.task_id]
    
    if task["status"] not in ["uploaded", "completed", "error"]:
        raise HTTPException(status_code=400, detail="Analysis already in progress")
    
    # Update status
    task["status"] = "analyzing"
    task["progress"] = 0.0
    task["mode"] = request.mode
    task["prompt"] = request.prompt
    
    # Run analysis in background
    background_tasks.add_task(run_analysis, request.task_id)
    
    return {
        "task_id": request.task_id,
        "status": "analyzing",
        "mode": request.mode,
    }


async def run_analysis(task_id: str):
    """Background task to run the analysis pipeline."""
    from ..pipelines import ChatModePipeline, GuidedModePipeline
    
    task = tasks[task_id]
    
    try:
        # Select pipeline based on mode
        if task.get("mode") == "guided":
            pipeline = GuidedModePipeline()
            state = await pipeline.run_with_approval(
                data=task["data"],
                user_prompt=task.get("prompt"),
            )
        else:
            pipeline = ChatModePipeline()
            state = await pipeline.analyze(
                data=task["data"],
                prompt=task.get("prompt"),
            )
        
        # Store results
        task["status"] = "completed"
        task["progress"] = 1.0
        task["state"] = state
        task["metrics"] = state.metrics
        task["feature_importance"] = state.explanations.get("feature_importance", {})
        task["best_model"] = max(
            state.metrics.items(),
            key=lambda x: x[1].get("f1_score", 0),
            default=("none", {})
        )[0] if state.metrics else None
        
        logger.info(f"Analysis completed: {task_id}")
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        task["status"] = "error"
        task["error"] = str(e)


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get status of an analysis task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    return TaskStatus(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", 0.0),
        current_stage=task.get("current_stage"),
        error=task.get("error"),
    )


@app.get("/results/{task_id}")
async def get_results(task_id: str):
    """Get results of a completed analysis."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not complete. Status: {task['status']}"
        )
    
    return AnalysisResult(
        task_id=task_id,
        status="completed",
        summary={
            "rows": task.get("rows", 0),
            "columns": task.get("columns", 0),
            "best_model": task.get("best_model"),
        },
        metrics=task.get("metrics", {}),
        feature_importance=task.get("feature_importance", {}),
        visualizations=[],  # Would contain chart URLs
    )


@app.post("/predict")
async def make_predictions(request: PredictionRequest):
    """Make predictions using trained model."""
    if request.task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[request.task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    try:
        # Get model from state
        state = task.get("state")
        if state is None or not state.models:
            raise HTTPException(status_code=400, detail="No trained model available")
        
        # Get best model
        best_model_name = task.get("best_model", "ensemble")
        model = state.models.get(best_model_name)
        
        if model is None:
            model = list(state.models.values())[0]
        
        # Make predictions
        df = pd.DataFrame(request.data)
        predictions = model.predict(df)
        
        # Try to get probabilities
        try:
            probabilities = model.predict_proba(df).tolist()
        except Exception:
            probabilities = None
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities,
            "model_used": best_model_name,
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("DataPilot AI Pro API started")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("DataPilot AI Pro API shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
