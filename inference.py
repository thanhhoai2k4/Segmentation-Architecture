import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.utils.files import increment_path  # Tiện ích của Ultralytics
from pathlib import Path
from IPython.display import Image
# --- 1. CẤU HÌNH ---

# ĐƯỜNG DẪN ĐẾN MODEL CỦA BẠN (quan trọng)
# Đây là model 'best.pt' hoặc 'yolov8n.pt'
YOLO_MODEL_PATH = "best.pt"

# ĐƯỜNG DẪN ĐẾN ẢNH LỚN CỦA BẠN (quan trọng)
# Ví dụ: ảnh 4K, ảnh drone...
LARGE_IMAGE_PATH = "F1_scaled.png"

# Kích thước ô (tile size) - Nên bằng kích thước huấn luyện của bạn
SLICE_HEIGHT = 640
SLICE_WIDTH = 640

# Tỷ lệ gối lên nhau (overlap)
OVERLAP_RATIO = 0.1

# Ngưỡng tin cậy (confidence threshold)
CONF_THRESHOLD = 0.3

# Nơi lưu kết quả
SAVE_DIR = Path("demo_data")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
save_path = str(increment_path(SAVE_DIR / Path(LARGE_IMAGE_PATH).name))

# --- 2. TẢI MODEL (thông qua SAHI) ---

print(f"Đang tải model từ: {YOLO_MODEL_PATH}")
try:
    # SAHI sẽ tự động nhận diện đây là model yolov8
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov11',
        model_path=YOLO_MODEL_PATH,
        confidence_threshold=CONF_THRESHOLD,
        device="cuda:0"  # "cuda:0" (GPU) hoặc "cpu"
    )
    print("Tải model thành công.")
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    exit()

# --- 3. CHẠY DỰ ĐOÁN "TILED" (Hàm chính) ---

print(f"Đang chạy dự đoán tiled trên: {LARGE_IMAGE_PATH}")
try:
    # Đây là hàm cốt lõi: nó tự động cắt, dự đoán trên từng ô, và ghép lại
    sahi_result = get_sliced_prediction(
        LARGE_IMAGE_PATH,
        detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO
    )

    print("Dự đoán Tiled hoàn tất.")

    # Lấy danh sách các đối tượng đã phát hiện
    object_prediction_list = sahi_result.object_prediction_list
    print(f"Tìm thấy {len(object_prediction_list)} đối tượng.")

    # --- 4. VẼ KẾT QUẢ VÀ LƯU ẢNH ---

    # Đọc ảnh gốc để vẽ
    image_to_draw = cv2.imread(LARGE_IMAGE_PATH)
    if image_to_draw is None:
        raise FileNotFoundError(f"Không thể đọc ảnh từ: {LARGE_IMAGE_PATH}")

    sahi_result.export_visuals(export_dir="demo_data/")
    Image("demo_data/prediction_visual.png")

except Exception as e:
    print(f"Lỗi trong quá trình dự đoán: {e}")