from huggingface_hub import HfApi, ModelFilter, ModelSearchArguments

api = HfApi()

# Get all valid search arguments
args = ModelSearchArguments()

# Using `ModelFilter` and `ModelSearchArguments` to find text classification in both PyTorch and TensorFlow
filt = ModelFilter(
    task=args.pipeline_tag.Text_to_Image,
    library=args.library.Diffusers,
    model_name="runwayml/stable-diffusion-inpainting"
)

print(api.list_models(filter=filt))
