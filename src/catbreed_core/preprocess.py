"""Tiền xử lý, chép nguyên từ notebooks/kaggle_runner.ipynb (mục 8.1 và read_rgb ở mục 1)."""
import io

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CONFIG = {"image_size": 224, "resize_size": 256}  # §2.4.2; khóa như CONFIG của notebook


def read_rgb(path):
    # Đọc header thực tế, kể cả PNG mang đuôi .jpg. Alpha ghép nền xám có quy định rõ.
    with Image.open(path) as source:
        source.load()
        if source.mode in ["RGBA", "LA"] or "transparency" in source.info:
            rgba = source.convert("RGBA")
            return Image.alpha_composite(
                Image.new("RGBA", rgba.size, (128, 128, 128, 255)), rgba
            ).convert("RGB")
        return source.convert("RGB")


def read_rgb_image(image):
    # Phần bọc: read_rgb của notebook nhận đường dẫn; ảnh PIL được ghi PNG không mất dữ liệu vào bộ nhớ
    # rồi đi qua đúng read_rgb (giữ nguyên quy tắc alpha → nền xám 128).
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return read_rgb(buffer)
    return read_rgb(image)


NORMALIZE = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
# Một kiểu nội suy chung cho cả bốn mô hình, kể cả EfficientNet-B0 (trọng số ImageNet của nó được công bố kèm
# bicubic): mọi mô hình nhận cùng một ảnh đầu vào nên chênh lệch kết quả là do mô hình, không do tiền xử lý.
INTERPOLATION = InterpolationMode.BILINEAR


def evaluation_geometry(interpolation=INTERPOLATION):
    # §2.4.2: cạnh ngắn 256 rồi cắt giữa 224; trimap dùng NEAREST (§2.8.2)
    return transforms.Compose(
        [
            transforms.Resize(CONFIG["resize_size"], interpolation=interpolation, antialias=True),
            transforms.CenterCrop(CONFIG["image_size"]),
        ]
    )


def make_transform(training=False, augmentation=True):
    # Chỉ nhánh đánh giá (training=False) được dùng khi suy luận; nhánh huấn luyện không chép.
    geometry = [evaluation_geometry()]
    return transforms.Compose(geometry + [transforms.ToTensor(), NORMALIZE])


def eval_transform():
    return make_transform()
