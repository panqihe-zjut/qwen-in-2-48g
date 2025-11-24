from qwenimage.QwenTextToImage import QwenImagePipeline as LocalQwenImagePipeline
from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"
cuda_list = ['cuda:0', 'cuda:1']

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = LocalQwenImagePipeline.from_pretrained(model_name, torch_dtype=torch_dtype)


pipe.text_encoder.to(cuda_list[0])
pipe.transformer.to(cuda_list[1])
pipe.vae.to(cuda_list[1])

# import pdb;pdb.set_trace()




positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}

# Generate image
prompt = '''PROMPT HERE'''

negative_prompt = " " # using an empty string if you do not have specific concept to remove


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1024, 1024),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["9:16"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example.png")
