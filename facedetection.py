from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def detect_faces(input_image_path: str):
    """
    Detects faces in the given image and returns the result.

    Parameters:
        input_image_path (str): The path to the image file.

    Returns:
        The detection results.
    """
    model_path = hf_hub_download(
        repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.onnx"
    )
    face_detection_model = YOLO(model_path, task="detect")
    detections = face_detection_model.predict(input_image_path, save=False)
    return detections
