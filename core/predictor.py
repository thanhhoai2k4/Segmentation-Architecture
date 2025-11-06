from ultralytics import YOLO
from PIL import Image

def predictor(image_path, model_name):
    # load model
    model = YOLO(model_name)

    image_source = image_path


    results = model(image_source)
    results[0].show()