import gdown
import zipfile
import os


def unzip(file : str = "dataset", extract_to_directory: str = "."):
    if os.path.exists(file):
        with zipfile.ZipFile(file, "r") as zip_ref:
            # giải nén
            zip_ref.extractall(extract_to_directory)
            print("Giải nén thành công")
    else:
        print("file không tồn tại. Yêu cầu tải d liệu.")

def download(link: str = "https://drive.usercontent.google.com/download?id=10jnA_l7ftsxjM29fdonPj68clDo8fr45&export=download&authuser=0"):
    """
        Tải dữ liệu từ Drive cho huấn luyện.
    :param link: Đường dẩn. Mặc định được ghi ở trên.
    :return: None
    """

    try:
        gdown.download(link)
        print("đã tải thành công")
    except Exception as e:
        print("Lổi thất bài: {}".format(e))