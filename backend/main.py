import os

from fastapi import FastAPI, Path
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from tasks.stable_diffusion_tasks import addTxt2ImgTask

# save your HF API token from https:/hf.co/settings/tokens
# as an env variable to avoid rate limiting
auth_token = os.getenv("auth_token")

app = FastAPI()

origins = [
    "http://localhost:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/api/txt2img",
    responses={200: {"content": {"image/png": {}}}},
)
def txt2img(
    prompt: str,
    negative_prompt: str,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 100,
    guidance_scale: float = 7.5,
):
    task_id = addTxt2ImgTask.delay(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    return Response(
        task_id,
        status_code=200,
    )


@app.get("/api/worker/tasks/{task_id}/status")
def getTaskStatus(task_id: str):
    task = addTxt2ImgTask.AsyncResult(task_id)

    return {
        "status": task.status,
        "result": task.result,
    }


@app.get("/api/hub/models/{model_name}")
async def getHubModels(model_name: str = Path(..., title="The model name to search for", example="runwayml/stable-diffusion-1-5")):
    from huggingface_hub import HfApi, ModelFilter, ModelSearchArguments

    api = HfApi()
    args = ModelSearchArguments()
    filt = ModelFilter(
        task=args.pipeline_tag.Text_to_Image,
        library=args.library.Diffusers,
        model_name=model_name,
    )

    return Response(api.list_models(filter=filt), status_code=200)
