"""
GCSE Automation API — FastAPI wrapper
Exposes the 4-stage pipeline as HTTP endpoints for the Next.js dashboard.
Designed to run on Railway (always-on, Python-native).

Endpoints:
  POST /pipeline/start     — upload PDFs, kick off background job
  GET  /pipeline/status/:id — poll job progress
  GET  /health             — Railway health check
"""

import os
import uuid
import asyncio
import tempfile
import traceback
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(title="GCSE Automation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to your Vercel URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# IN-MEMORY JOB STORE
# Simple dict — persists for the lifetime of the Railway process.
# Good enough for a content team. Replace with Redis if you scale.
# ─────────────────────────────────────────────

jobs: dict[str, dict] = {}


def new_job(paper_label: str) -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id":          job_id,
        "paper_label": paper_label,
        "status":      "queued",       # queued | running | done | failed
        "stage":       0,              # 0–4
        "stage_label": "Queued",
        "progress":    0,              # 0–100
        "rows_total":  0,
        "rows_done":   0,
        "error":       None,
        "started_at":  datetime.utcnow().isoformat(),
        "finished_at": None,
        "sheet_url":   f"https://docs.google.com/spreadsheets/d/{os.getenv('GSHEET_ID')}",
    }
    return job_id


def update_job(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)


# ─────────────────────────────────────────────
# PIPELINE RUNNER (background task)
# ─────────────────────────────────────────────

async def run_pipeline(
    job_id: str,
    question_pdf_path: str,
    marking_pdf_path: str,
    paper_label: str,
):
    """
    Runs all 4 stages in sequence in a background task.
    Updates job state at each stage so the dashboard can poll progress.
    """
    try:
        update_job(job_id, status="running")

        # ── Stage 1: Extract ──────────────────────────────────────────────
        update_job(job_id, stage=1, stage_label="Extracting questions from PDF", progress=5)

        from extractor import extract_pdf
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, lambda: extract_pdf(
            question_pdf_path=question_pdf_path,
            marking_pdf_path=marking_pdf_path,
            paper_label=paper_label,
            output_dir=tempfile.mkdtemp(prefix="gcse_figures_"),
            upload_to_drive=True,
            write_to_sheet=True,
        ))

        update_job(job_id, rows_total=len(rows), progress=25,
                   stage_label=f"Extracted {len(rows)} questions")

        # ── Stage 2: Rewrite ─────────────────────────────────────────────
        update_job(job_id, stage=2, stage_label="Rewriting questions", progress=30)

        from rewriter import run_rewriter
        await loop.run_in_executor(None, lambda: run_rewriter(overwrite=False))

        update_job(job_id, progress=50, stage_label="Questions rewritten")

        # ── Stage 3: Generate prompts ────────────────────────────────────
        update_job(job_id, stage=3, stage_label="Generating evaluation prompts", progress=55)

        from prompt_generator import run_prompt_generator
        await loop.run_in_executor(None, lambda: run_prompt_generator(overwrite=False))

        update_job(job_id, progress=70, stage_label="Prompts generated")

        # ── Stage 4: Test ────────────────────────────────────────────────
        update_job(job_id, stage=4, stage_label="Running automated tests", progress=75)

        from tester import run_tester
        await loop.run_in_executor(None, lambda: run_tester(overwrite=False))

        # ── Done ─────────────────────────────────────────────────────────
        update_job(
            job_id,
            status="done",
            stage=4,
            stage_label="Complete — ready for review",
            progress=100,
            finished_at=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        update_job(
            job_id,
            status="failed",
            stage_label="Pipeline failed",
            error=traceback.format_exc(),
            finished_at=datetime.utcnow().isoformat(),
        )
        raise

    finally:
        # Clean up temp PDF files
        for path in [question_pdf_path, marking_pdf_path]:
            try:
                os.remove(path)
            except Exception:
                pass


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Railway health check."""
    return {"status": "ok"}


@app.post("/pipeline/start")
async def start_pipeline(
    background_tasks: BackgroundTasks,
    question_pdf: UploadFile = File(...),
    marking_pdf:  UploadFile = File(...),
    paper_label:  str        = Form(...),
):
    """
    Accept two PDF uploads and a paper label, then kick off the pipeline.
    Returns a job_id immediately — poll /pipeline/status/:id for progress.
    """

    # Validate file types
    for f in [question_pdf, marking_pdf]:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF")

    # Save uploads to temp files
    tmp_dir = tempfile.mkdtemp(prefix="gcse_upload_")

    q_path = os.path.join(tmp_dir, "question_paper.pdf")
    with open(q_path, "wb") as out:
        out.write(await question_pdf.read())

    m_path = os.path.join(tmp_dir, "marking_scheme.pdf")
    with open(m_path, "wb") as out:
        out.write(await marking_pdf.read())

    # Create job and start background pipeline
    job_id = new_job(paper_label)
    background_tasks.add_task(
        run_pipeline,
        job_id=job_id,
        question_pdf_path=q_path,
        marking_pdf_path=m_path,
        paper_label=paper_label,
    )

    return {
        "job_id":    job_id,
        "status":    "queued",
        "message":   f"Pipeline started for '{paper_label}'",
        "poll_url":  f"/pipeline/status/{job_id}",
    }


@app.get("/pipeline/status/{job_id}")
def get_status(job_id: str):
    """
    Poll this endpoint to get pipeline progress.
    Returns stage, progress %, rows processed, and error if any.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/pipeline/jobs")
def list_jobs():
    """List all jobs — useful for the dashboard to show recent uploads."""
    return list(reversed(list(jobs.values())))