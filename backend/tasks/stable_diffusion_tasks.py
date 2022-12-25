import io

from tasks.worker import celery
from pipelines.diffuser import Diffuser

diffUser = Diffuser("runwayml/stable-diffusion-v1-5")


@celery.task(name="aiartist.tasks.txt2img.add")
def addTxt2ImgTask(
    prompt, negative_prompt, height, width, num_inference_steps, guidance_scale
):
    image = diffUser.txt2img(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    with io.BytesIO() as output:
        image.save(output, format="PNG")

    return output.getvalue()
