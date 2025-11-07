# Detection-Architecture
Image processing and recognition of architectural objects before importing them into Revit.

# 1. Download data.

[Bộ dữ liệu kiến trúc nhà:](https://www.kaggle.com/datasets/mthcknng/my-dataset-floor-plan): Đây là bộ dữ liệu tôi đả trích xuất từ [cubicassa5k](https://github.com/CubiCasa/CubiCasa5k.git)
. Lấy tất cả ảnh từ folder: high_quality_architectural. Sau đó trích xuất các nhãn từ file vsg tương ứng. Trong dự án này tôi sử dụng 3 đối tượng:

- Wall
- Door
- Window

Dữ liệu được tôi cấu trúc theo chuẩn của YOLO Segmentation: Dùng đa giác để vẽ mặt ná (mask) bao quan vật thể.

Cấu trúc:

```commandline
/dataset_folder
|
|-- /images
|   |-- /train
|   |   |-- img1.jpg
|   |   |-- img2.jpg
|   |   `-- ...
|   `-- /val
|       |-- img101.jpg
|       `-- ...
|
|-- /labels
|   |-- /train
|   |   |-- img1.txt
|   |   |-- img2.txt
|   |   `-- ...
|   `-- /val
|       |-- img101.txt
|       `-- ...
|
`-- data.yaml

```

Note:
- Chất lượng ảnh lớn: 2000x2000. Vì thế khi đi qua model sẽ được scale lại thành 640x640=> các object trở nên nhỏ bé và dể bị miss. Vì thế tôi đả cắt ảnh thành nhiều tấm ảnh nhỏ.

# 2. Kiến trúc model

Tôi sử dụng yolo segmnet của ultralytic để nhận dạng và phân vùng các đối tượng.
Ở đây tôi sử dụng phiên bản nhẹ nhất của v11 để đào tạo và quấn luyện model. 

3. kết quả




image1

image2


# 4. Đóng góp