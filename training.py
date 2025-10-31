from ultralytics import YOLO
from dowload_data import *
from config import *


# download dữ liêu
# url = "https://drive.google.com/file/d/10jnA_l7ftsxjM29fdonPj68clDo8fr45/view?usp=sharing"
# download(url)

if os.path.isfile("dataset.zip"):
    print("đả tải dữ liệu thành công.")

# unzip
file = "dataset.rar"
extract_to_directory = "."
unzip(
    file,
    extract_to_directory)
if os.path.isfile("dataset"):
    print("Đã giải nén dữ liệu và tồn tại dataset.")


# huấn luyện YOLO lại từ ban đầu không sử dung pre.
from ultralytics import YOLO

# Load a model
model = YOLO(model)  # build a new model from YAML
# Train the model
results = model.train(
    data=data, epochs=epochs, imgsz=640, batch=batch, plots=plots,
)