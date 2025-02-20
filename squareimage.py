from PIL import Image
from facedetection import detect_faces


def crop_image_to_square(input_image_path, output_image_path):
    with Image.open(input_image_path) as img:
        width, height = img.size

        face_detection_results = detect_faces(input_image_path)
        bounding_boxes = face_detection_results[0].boxes.xyxy

        if len(bounding_boxes) != 1:
            raise ValueError("This program only supports images with a single face.")

        bounding_box = [float(x) for x in bounding_boxes[0]]
        face_center_x = (bounding_box[0] + bounding_box[2]) / 2
        face_center_y = (bounding_box[1] + bounding_box[3]) / 2

        square_crop_size = min(width, height)
        crop_left = face_center_x - square_crop_size / 2
        crop_top = face_center_y - square_crop_size / 2

        crop_left = max(0, min(crop_left, width - square_crop_size))
        crop_top = max(0, min(crop_top, height - square_crop_size))
        crop_right = crop_left + square_crop_size
        crop_bottom = crop_top + square_crop_size

        cropped_image = img.crop((crop_left, crop_top, crop_right, crop_bottom))
        cropped_image.save(output_image_path)
