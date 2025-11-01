import yaml
from ultralytics import YOLO


class YOLOV11Trainer:
    """

        Đây là 1 class để đóng gói quá trình huấn luyện model.
        YOLO SEGMENT V11: Phiên bản: extra large (x)
        => độ chính xác cao.

    """
    def __init__(self, config_path: str):
        self.config_path = config_path # đường dẩn đến file yaml
        self.config = self.load_config()


    def load_config(self) -> dict:
        """

            Tải file cấu hình yaml từ config.

        """

        try:
            # Mở file
            with open(self.config_path, "r") as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
                print("_____ TẢI THÀNH CÔNG ______")
                return config

        except FileNotFoundError:
            print("Không tìm thấy file cấu hình tại {}".format(self.config_path))
            return None

        except Exception as e:
            print("Lỗi ngoài khi đoc file cấu hình: {}".format(e))
            return None

    def train(self):
        """

            bắt đầu huấn luyện ở đây.

            1. Cấu hình
            2. huấn luyện

        """

        # kiem tra config
        if  not self.config:
            print("_____ Huấn Luyện thất bại do không có cấu hình _____")
            return

        # tìm model_name trong config. Nếu không thầy thì default: YOLO11x-seg.pt
        model_name = self.config.get("model_name","yolo11x-seg.yaml")
        print("_____ Đang ta mô hình cơ sở: {}".format(model_name))
        # load model
        model = YOLO(model_name) # load model với phiên bản nặng nhất.

        dataset_path = self.config.get('dataset_yaml')
        training_params = self.config.get('training_params', {})

        if not dataset_path:
            print("train_config.yaml chưa được định nghĩa.")
            return

        print("---- Bắt đầu huấn luyện ----")
        print("Dataset path: {}".format(dataset_path))
        print("Các tham số: {}".format(training_params))


        # 2. training model YOLO SEGMENT 11 extra large (x)
        model.train(
            data=dataset_path,
            **training_params
        )

        print("---- Quá trình huấn luyện thành công ----")