import argparse
import os
import uuid
from typing import Dict, Literal, Optional

import torch
import uvicorn
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

app = FastAPI(
    title="wan-api",
    version="0.1.0",
    servers=[{"url": "http://10.31.28.2:50002", "description": "Development Environment"}],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VideoSynthesisRequest(BaseModel):
    prompt: str = Field("A cat and a dog baking a cake together in a kitchen.", description="The prompt to be encoded")
    negative_prompt: Optional[str] = Field("Bright tones", description="The prompt not to guide the image generation")
    num_frames: int = Field(33, description="The number of frames in the generated video")
    fps: int = Field(16, description="Frame per second, leading to the result video length")
    width: int = Field(624, description="The width in pixels of the generated frame")
    height: int = Field(624, description="The height in pixels of the generated frame")
    num_inference_steps: Optional[int] = Field(10, description="The number of denoising steps")


class TaskStatus(BaseModel):
    task_id: str = Field(description="The video synthesis task id")
    status: Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED"] = Field(description="Task status")
    message: Optional[str] = Field(None, description="Task message")


tasks: Dict[str, TaskStatus] = {}


def inference(params: VideoSynthesisRequest, task_id: str):
    tasks[task_id].status = "RUNNING"
    try:
        frames = pipe(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            num_frames=params.num_frames,
            width=params.width,
            height=params.height,
            num_inference_steps=params.num_inference_steps,
        ).frames[0]
        export_to_video(frames, f"./temp/{task_id}.mp4", fps=params.fps)
        tasks[task_id].status = "SUCCEEDED"
    except Exception as e:
        tasks[task_id].status = "FAILED"
        tasks[task_id].message = e


@app.post("/api/v1/video-synthesis/gen", tags=["Video Synthesis"])
async def submit_a_generation_task(req: VideoSynthesisRequest, background_tasks: BackgroundTasks) -> TaskStatus:
    task_id = str(uuid.uuid4())
    background_tasks.add_task(inference, params=req, task_id=task_id)
    tasks[task_id] = TaskStatus(task_id=task_id, status="PENDING")
    return tasks[task_id]


@app.get("/api/v1/video-synthesis/tasks/{task_id}", tags=["Video Synthesis"])
async def get_task_status(task_id: str) -> TaskStatus:
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.get("/api/v1/video-synthesis/result/{task_id}", tags=["Video Synthesis"])
async def get_video_result(task_id: str):
    """Return the video file and clear corresponding task info and video"""
    file_path = f"./temp/{task_id}.mp4"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video not found")
    if task_id in tasks:
        del tasks[task_id]
    return FileResponse(path=file_path, media_type="video/mp4", background=BackgroundTask(lambda: os.remove(file_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50002)
    parser.add_argument("--path", type=str, default="pretrained_models/Wan2.1-T2V-1.3B-Diffusers")
    args = parser.parse_args()

    pipe = WanPipeline.from_pretrained(args.path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
