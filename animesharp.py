import onnxruntime as ort
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download


def upscale_anime_image(input_image_path, output_image_path):
    """
    Upscales an anime-style image using AnimeSharp model

    Args:
        input_image_path (str): Path to input image
        output_image_path (str): Path to save upscaled image
    """
    # Download the model if not already present
    model_path = hf_hub_download(
        repo_id="Kim2091/2x-AnimeSharpV4",
        filename="2x-AnimeSharpV4_Fast_RCAN_PU_fp16_opset17.onnx",
        revision="main",
    )

    # Load and prepare the image
    img = Image.open(input_image_path)
    img_np = np.array(img).astype(np.float32)
    img_np = img_np.transpose(2, 0, 1)  # HWC to CHW
    img_np = np.expand_dims(img_np, 0)  # Add batch dimension
    img_np = img_np / 255.0  # Normalize to [0, 1]

    # Convert to float16 for the model
    img_np = img_np.astype(np.float16)

    # Create inference session with CUDA provider if available
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(None, {input_name: img_np})
    output_img = outputs[0][0]

    # Post-process the output
    output_img = output_img.transpose(1, 2, 0)  # CHW to HWC
    output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)

    # Save the upscaled image
    Image.fromarray(output_img).save(output_image_path)
