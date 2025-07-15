#!/usr/bin/env python3
"""
FastAPI Web Interface for Developer Tool Testing Pipeline

Modern web interface to configure, run, and monitor the complete testing pipeline.
Provides real-time progress updates and comprehensive results display.
"""

from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel
from pathlib import Path

# Import our pipeline
from agents.test import DeveloperToolTestingPipeline, PipelineConfig

app = FastAPI(title="Developer Tool Testing Pipeline", version="1.0.0")

# Templates and static files - use package relative path
template_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(template_dir))

# In-memory storage for pipeline runs (in production, use Redis/DB)
pipeline_runs: Dict[str, Dict] = {}

class PipelineRequest(BaseModel):
    tool_name: str
    base_url: str
    api_keys: Dict[str, str] = {}
    urls_to_exclude: list = []
    max_pages: int = 20
    max_depth: int = 3
    keywords: list = ["api", "documentation", "guide"]
    max_workers: int = 8

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page with pipeline configuration form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/start-pipeline")
async def start_pipeline(background_tasks: BackgroundTasks, request: PipelineRequest):
    """Start a new pipeline run"""
    
    # Generate unique run ID
    run_id = str(uuid.uuid4())
    
    # Store initial run info
    pipeline_runs[run_id] = {
        "id": run_id,
        "status": "starting",
        "start_time": datetime.now().isoformat(),
        "config": request.dict(),
        "progress": {"current_stage": "initializing", "pages_processed": 0, "total_pages": 0},
        "results": {},
        "error": None
    }
    
    # Start pipeline in background
    background_tasks.add_task(run_pipeline_async, run_id, request)
    
    return {"run_id": run_id, "status": "started"}

@app.get("/status/{run_id}")
async def get_pipeline_status(run_id: str):
    """Get current status of a pipeline run"""
    if run_id not in pipeline_runs:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    return pipeline_runs[run_id]

@app.get("/results/{run_id}", response_class=HTMLResponse)
async def view_results(request: Request, run_id: str):
    """View detailed results of a pipeline run"""
    if run_id not in pipeline_runs:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    run_data = pipeline_runs[run_id]
    return templates.TemplateResponse("results.html", {
        "request": request,
        "run_id": run_id,
        "run_data": run_data
    })

@app.get("/api/runs")
async def list_runs():
    """List all pipeline runs"""
    return {"runs": list(pipeline_runs.values())}

async def run_pipeline_async(run_id: str, request: PipelineRequest):
    """Run the pipeline asynchronously and update status"""
    
    try:
        # Update status
        pipeline_runs[run_id]["status"] = "running"
        
        # Create pipeline config
        config = PipelineConfig(
            tool_name=request.tool_name,
            base_url=request.base_url,
            api_keys=request.api_keys,
            urls_to_exclude=request.urls_to_exclude,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            keywords=request.keywords,
            max_workers=request.max_workers
        )
        
        # Run pipeline
        pipeline = DeveloperToolTestingPipeline(config)
        result = pipeline.run_complete_pipeline()
        
        # Store results
        pipeline_runs[run_id].update({
            "status": "completed" if result.current_stage == "completed" else "failed",
            "end_time": datetime.now().isoformat(),
            "results": {
                "state": result.model_dump(),
                "total_pages": result.total_pages,
                "completed_pages": result.completed_pages,
                "failed_pages": result.failed_pages,
                "overall_report": result.overall_report.model_dump() if result.overall_report else None,
                "page_reports": {}
            }
        })
        
        # Extract page reports
        for url, page in result.pages.items():
            if page.page_report:
                pipeline_runs[run_id]["results"]["page_reports"][url] = page.page_report.model_dump()
        
    except Exception as e:
        pipeline_runs[run_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": str(e)
        })

def main():
    """Main entry point for the web interface"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)

if __name__ == "__main__":
    main()
