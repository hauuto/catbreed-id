"""Mô-đun tiền xử lý ảnh suy luận chuẩn dùng chung (Tuần 4).

Theo Đề cương đồ án (Mục 5.2):
- Quy trình: convert("RGB") -> exif_transpose -> resize cạnh ngắn 256 -> crop giữa 224x224
  -> chia 255.0 -> chuẩn hóa ImageNet (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  -> định dạng CHW float32.

Dùng PIL, NumPy và torchvision. Hàm này xử lý EXIF; các luồng đánh giá
và dịch vụ hiện không xoay EXIF, nên ảnh có EXIF cần được đối chiếu riêng.
"""
from typing import Tuple, Union
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import functional as TVF, InterpolationMode

# Các hằng số tiền xử lý chuẩn duy nhất cho toàn bộ dự án
IMG_SIZE: int = 224
MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def preprocess_pil(img: Image.Image) -> Image.Image:
    """Tiền xử lý ảnh dạng PIL Image phục vụ hiển thị hoặc các bước xử lý ảnh không cần tensor.

    - Xoay ảnh theo cờ EXIF (nếu có, tránh trường hợp ảnh điện thoại bị quay ngang/ngược).
    - Chuyển hệ màu RGB (loại bỏ kênh alpha của PNG, chuyển ảnh đơn kênh xám sang 3 kênh).
    - Resize cạnh ngắn 256 bằng nội suy song tuyến, rồi crop giữa 224x224.
    """
    # Xử lý EXIF orientation nếu có
    img = ImageOps.exif_transpose(img)
    # Chuẩn hóa về RGB
    img = img.convert("RGB")
    return TVF.center_crop(TVF.resize(img, 256, interpolation=InterpolationMode.BILINEAR, antialias=True), [IMG_SIZE, IMG_SIZE])


def preprocess_image(
    img: Image.Image,
    add_batch_dim: bool = False,
) -> np.ndarray:
    """Tiền xử lý ảnh thành mảng NumPy chuẩn hóa ImageNet.

    Args:
        img: Đối tượng PIL Image đầu vào.
        add_batch_dim: Nếu True, trả về mảng shape (1, 3, 224, 224); nếu False, trả về (3, 224, 224).

    Returns:
        np.ndarray: Mảng float32 đã chuẩn hóa theo ImageNet, layout CHW (hoặc NCHW).
    """
    pil_resized = preprocess_pil(img)

    # Chuyển thành mảng float32 trong khoảng [0.0, 1.0], shape (H, W, C)
    arr = np.asarray(pil_resized, dtype=np.float32) / 255.0

    # Chuẩn hóa theo ImageNet: (x - mean) / std
    mean_arr = np.array(MEAN, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(STD, dtype=np.float32).reshape(1, 1, 3)
    normalized = (arr - mean_arr) / std_arr

    # Chuyển từ HWC (224, 224, 3) sang CHW (3, 224, 224)
    chw = np.transpose(normalized, (2, 0, 1))

    if add_batch_dim:
        return np.expand_dims(chw, axis=0)
    return chw

