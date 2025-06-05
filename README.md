# Uwuifier

Uwuifier is a Python project that uses stable diffusion to synthesise anime-style profiles pictures.

It first processes images by first cropping them to a square focusing on a face (using [facedetection.py](facedetection.py) and [squareimage.py](squareimage.py)), then generating an anime-style version of the image with [animesd.py](animesd.py) and finally upscaling the result using [animesharp.py](animesharp.py).

## Features

- **Face Detection & Cropping:** Uses a YOLO-based face detector to crop the input image around the detected face.
- **Anime Image Generation:** Applies an anime style transformation with Stable Diffusion and ControlNet.
- **Upscaling:** Enhances the generated anime image using an ONNX model.

## Requirements (I think)

- Python 3.10+  
- [Pillow](https://python-pillow.org/)
- [torch](https://pytorch.org/)
- [ultralytics](https://github.com/ultralytics/ultralytics)
- [diffusers](https://github.com/huggingface/diffusers)
- [transformers](https://github.com/huggingface/transformers)
- [onnxruntime](https://github.com/microsoft/onnxruntime)
- [huggingface_hub](https://github.com/huggingface/huggingface_hub)
- [controlnet_aux](path/to/file)

Additional dependencies may be required. Make sure to check each module's documentation for installation instructions.
