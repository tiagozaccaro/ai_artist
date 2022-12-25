from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
import torch


class Diffuser:
    def __init__(self, model_id):
        self._model_id = None
        self._txt2img_pipe = None
        self._img2img_pipe = None
        self._inpaint_pipe = None

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.backends.cudnn.benchmark = True

        if self._device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True

        if model_id is not None:
            self.loadModel(model_id)
            self._model_id = model_id

    def loadModel(self, model_id):
        # Prevent reloading the same model
        if self._model_id == model_id:
            return

        # Delete the old model
        if self._txt2img_pipe is not None:
            del self._txt2img_pipe

        # Load the new model
        self._txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            revision="fp16",
            cache_dir="models",
            safety_checker=None,
        )

        self._txt2img_pipe.progress_bar

        self._txt2img_pipe = self._txt2img_pipe.to(self._device)

        self._txt2img_pipe.enable_sequential_cpu_offload()
        self._txt2img_pipe.enable_attention_slicing()
        # pipe.enable_xformers_memory_efficient_attention()
        self._txt2img_pipe.enable_vae_slicing()

        self._img2img_pipe = StableDiffusionImg2ImgPipeline(
            **self._txt2img_pipe.components
        )

        self._inpaint_pipe = StableDiffusionInpaintPipeline(
            **self._txt2img_pipe.components
        )

        self._model_id = model_id

    def txt2img(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        num_inference_steps,
        guidance_scale,
    ):
        return self._txt2img_pipe(
            prompt,
            negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
