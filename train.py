import argparse
from core import YOLOV11Trainer


def main():


    # Tạo 1 trình cú pháp đối số dòng lệnh
    parser = argparse.ArgumentParser(description="TRAINING MODEL YOLO SEGMENT")

    # + đối số
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",

    )

    args = parser.parse_args()

    # training
    trainer = YOLOV11Trainer(
        config_path=args.config
    )
    trainer.train()

if __name__ == "__main__":
    main()