import os
import torch
from diffusers.utils import load_image
from controlnet_aux import LineartAnimeDetector
from transformers import CLIPTextModel
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


def generate_anime_image(
    input_image_path: str,
    output_dir: str,
    output_image_path: str,
    prompt: str,
    seed: int,
    negative_prompt: str = "",
):
    """
    Generates an image using Stable Diffusion with ControlNet.

    Parameters:
    - input_image_path (str): URL or file path of the input image.
    - output_dir (str): Directory where the output images will be saved.
    - prompt (str): Text prompt to guide image generation.
    - seed (int): Random seed for reproducibility.
    - negative_prompt (str): Text prompt to guide negative aspects of image generation.

    Returns:
    - The generated PIL image.
    """
    input_pil_image = load_image(input_image_path)
    input_pil_image = input_pil_image.resize((512, 512))

    lineart_anime_processor = LineartAnimeDetector.from_pretrained(
        "lllyasviel/Annotators"
    )
    lineart_control_image = lineart_anime_processor(input_pil_image)
    os.makedirs(output_dir, exist_ok=True)
    lineart_control_image.save(os.path.join(output_dir, "control.png"))

    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder", num_hidden_layers=11
    )

    lineart_checkpoint = "lllyasviel/control_v11p_sd15s2_lineart_anime"
    lineart_controlnet = ControlNetModel.from_pretrained(lineart_checkpoint)
    diffusion_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        text_encoder=text_encoder,
        controlnet=lineart_controlnet,
    )
    diffusion_pipeline.scheduler = UniPCMultistepScheduler.from_config(
        diffusion_pipeline.scheduler.config
    )
    diffusion_pipeline.to("cpu")

    torch_generator = torch.manual_seed(seed)
    final_anime_image = diffusion_pipeline(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        generator=torch_generator,
        image=lineart_control_image,
    ).images[0]

    final_anime_image.save(os.path.join(output_dir, output_image_path))
    return final_anime_image
