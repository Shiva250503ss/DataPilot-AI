"""
DataPilot AI Pro - FastAPI Backend
===================================
REST API for the data science platform.

Endpoints:
- POST /upload - Upload dataset for analysis
- POST /analyze - Start analysis pipeline
- GET /status/{task_id} - Check task status
- GET /results/{task_id} - Get analysis results
- POST /predict - Make predictions with trained model
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
    
    Accepts CSV files and returns a task_id for tracking.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read file content
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        
        # Create task
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": "uploaded",
            "progress": 0.0,
            "data": df,
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
        }
        
        logger.info(f"File uploaded: {file.filename} ({len(df)} rows)")
        
        return {
            "task_id": task_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
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
