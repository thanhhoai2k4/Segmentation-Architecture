from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os
import json
import csv
from IPython.display import Image, display


def predictor(
        image_path: str,
        model_path: str,
        json_path: str,
        Slice_height: int = 640,
        Slice_width: int = 640,
        OVERLAP_RATIO: float = 0.2,
        CONF_THRESHOLD: float = 0.3,
)-> None:


    assert os.path.exists(image_path), "Image File  does not exist"
    assert os.path.exists(model_path), "Model File  does not exist"
    assert isinstance(Slice_height, int), "Slice Height must be an integer"
    assert isinstance(Slice_width, int), "Slice Width must be an integer"
    assert (OVERLAP_RATIO > 0 and OVERLAP_RATIO <1), "Overlap ratio must be >= 0"
    assert  CONF_THRESHOLD > 0 and CONF_THRESHOLD < 1, "Confusion threshold must be >= 1"
    assert os.path.exists(json_path), "Json folder  does not exist"

    SAVE_DIR = "my_result/result"
    name_image = os.path.splitext(os.path.basename(image_path))[0]
    path_json_file = os.path.join(json_path, name_image + ".json")
    path_csv_file = os.path.join(json_path, name_image + ".csv")


    # load model
    # https://docs.ultralytics.com/guides/sahi-tiled-inference/#import-modules-and-download-resources
    # Instantiate the Model
    # Khoi tao model
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=0.3,
            device="cpu",  # or 'cuda:0'
        )
        print("Load access model.")
    except:
        print("check path of model.")
        return

    # predict.
    # https://docs.ultralytics.com/guides/sahi-tiled-inference/#sliced-inference-with-yolo11
    # Sliced Inference with YOLO11

    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=Slice_height,
        slice_width=Slice_width,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=CONF_THRESHOLD,
    )

    # convert to list
    object_prediction_list = result.object_prediction_list
    print("len(object_prediction_list):", len(object_prediction_list))

    # expert json file
    json_result = list()
    for pred in object_prediction_list:
        # Get xyxy
        bbox = pred.bbox.to_xyxy()

        # get segment:
        segmentation_polygon = pred.mask.segmentation[0]

        json_result.append({
            "category_name": pred.category.name,
            "score": float(pred.score.value),
            "bbox_xyxy": [float(coord) for coord in bbox],
            "segmentation_polygon": [float(point) for point in segmentation_polygon]
        })

    # write to json
    with open(path_json_file, "w", encoding="utf-8") as f:
        json.dump(json_result, f, indent=4, ensure_ascii=False)

    print("export success json.")


    # tao file csv
    with open(path_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Viết header
        writer.writerow(["category", "score", "x1", "y1", "x2", "y2"])

        # Viết dữ liệu
        for pred in object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            writer.writerow([
                pred.category.name,
                float(pred.score.value),
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3])
            ])

    print("Export CSV successfully")


    # save anh.
    result.export_visuals(export_dir="my_result/result")
    Image("my_result/result/1.png")