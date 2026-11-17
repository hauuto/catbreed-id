"""Dịch vụ FastAPI - Hợp đồng API Mô hình & Dịch vụ (Tuần 3 - Phiên bản 1.0).

Cung cấp điểm cuối phân loại giống mèo, kiểm chứng từ chối Out-of-Distribution (OOD)
và chuẩn bị tích hợp Grad-CAM giải thích quyết định mạng nơ-ron.

Chạy từ thư mục gốc của dự án:
    pip install -r app/api/requirements.txt
    python -m uvicorn app.api.main:app --reload --port 8000
Tài liệu tự động (Swagger UI): http://127.0.0.1:8000/docs
"""
import io
import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .temp_model import BaseCatClassifier, MockCatClassifier

# Cấu hình giới hạn
MAX_BYTES = 8 * 1024 * 1024  # 8 MB
ALLOWED_TYPES = {"image/jpeg", "image/png"}

# ---------------------------------------------------------------------------
# Pydantic Schemas - Định nghĩa Hợp đồng dữ liệu JSON v1.0
# ---------------------------------------------------------------------------

class Top5Item(BaseModel):
    label: str = Field(..., description="Tên giống mèo trong 12 giống mục tiêu")
    prob: float = Field(..., ge=0.0, le=1.0, description="Xác suất dự đoán tương ứng [0, 1]")


class OODItem(BaseModel):
    method: str = Field(..., description="Tên phương pháp OOD (MSP, MaxLogit, Energy, Mahalanobis, KNN)")
    score: float = Field(..., description="Điểm số đo được. Quy ước: càng cao càng giống In-Distribution (mèo)")
    threshold: Optional[float] = Field(None, description="Ngưỡng quyết định; null khi chưa chốt")
    accept: Optional[bool] = Field(None, description="Quyết định OOD; null khi chưa chốt ngưỡng")


class ModelMetadata(BaseModel):
    backbone: str = Field(..., description="Kiến trúc mạng nơ-ron nền tảng (MobileNetV2, EfficientNet-B0)")
    mode: str = Field(..., description="Chế độ mô hình: feature_extractor hoặc fine_tuning")
    seed: int = Field(..., description="Hạt giống ngẫu nhiên đảm bảo tính tái lập")
    checkpoint_id: str = Field(..., description="Mã định danh phiên bản checkpoint hoặc commit hash")


class PredictResponse(BaseModel):
    api_version: str = Field(default="1.0", description="Phiên bản định dạng API")
    status: str = Field(..., description="accepted, rejected hoặc unverified khi chưa có ngưỡng OOD")
    top_label: Optional[str] = Field(None, description="Tên giống mèo có xác suất cao nhất (null khi rejected)")
    top5: List[Top5Item] = Field(..., min_length=1, max_length=12, description="Top giống mèo có xác suất cao nhất, sắp xếp giảm dần")
    ood: Optional[OODItem] = Field(None, description="Kết quả OOD chính; null khi chưa hiệu chỉnh")
    ood_all: Optional[List[OODItem]] = Field(None, description="Kết quả đầy đủ cả 5 phương pháp OOD cho chế độ nâng cao")
    model: ModelMetadata = Field(..., description="Thông tin siêu dữ liệu về mô hình đang chạy")
    heatmap: Optional[str] = Field(None, description="Bản đồ nhiệt Grad-CAM mã hóa Base64 PNG (dành cho tuần 12)")
    cam_input: Optional[str] = Field(None, description="Ảnh đầu vào mô hình 224x224 mã hóa Base64 PNG để hiển thị Grad-CAM")
    latency_ms: int = Field(0, description="Thời gian trễ xử lý suy luận và tính toán (mili-giây)")
    is_mock: bool = Field(True, description="True nếu kết quả từ mô hình tạm giả lập")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Trạng thái dịch vụ")
    api_version: str = Field(default="1.0", description="Phiên bản API")
    is_mock: bool = Field(..., description="True nếu dịch vụ đang dùng mô hình tạm")
    num_classes: int = Field(..., description="Số lượng giống mèo hỗ trợ nhận diện")
    classes: List[str] = Field(..., description="Danh sách 12 giống mèo mục tiêu")


# ---------------------------------------------------------------------------
# Khởi tạo Ứng dụng & Mô hình
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CatBreedID API",
    description=(
        "Hợp đồng API mô hình – dịch vụ (v1.0) cho đồ án Nhận diện giống mèo "
        "kết hợp kiểm chứng từ chối Out-of-Distribution (OOD) và giải thích Grad-CAM."
    ),
    version="1.0.0",
)

# Cho phép CORS từ mọi origin để giao diện (Live Server, file://, localhost) gọi sang dễ dàng
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chạy mô hình thật mặc định; chỉ dùng mô hình giả khi được chọn rõ ràng.
model_mode = os.environ.get("CATBREED_MODEL", "real")
if model_mode == "mock":
    classifier: BaseCatClassifier = MockCatClassifier()
elif model_mode == "real":
    from .real_model import ResNet50Classifier
    checkpoint_path = Path(__file__).resolve().parents[2] / "model_package" / "weights.pth"
    classifier = ResNet50Classifier(checkpoint_path)
else:
    raise ValueError("CATBREED_MODEL phải là real hoặc mock")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Kiểm tra trạng thái hoạt động của dịch vụ",
    tags=["Hệ thống"],
)
def health():
    return {
        "status": "ok",
        "api_version": "1.0",
        "is_mock": getattr(classifier, "is_mock", True),
        "num_classes": len(classifier.classes),
        "classes": classifier.classes,
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Phân tích ảnh giống mèo và kiểm chứng OOD",
    description="Nhận một file ảnh (JPG/PNG <= 8MB), trả về dự đoán phân loại giống mèo và điểm từ chối OOD.",
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "Không đọc được dữ liệu ảnh (file hỏng hoặc không đúng định dạng pixel)."},
        413: {"description": "Dung lượng ảnh vượt quá giới hạn 8 MB."},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"description": "Định dạng file không được hỗ trợ (chỉ nhận JPG hoặc PNG)."},
    },
    tags=["Nhận diện & OOD"],
)
async def predict_endpoint(file: UploadFile = File(..., description="File ảnh JPG hoặc PNG")):
    # 1. Kiểm tra loại file
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Chỉ chấp nhận ảnh JPG hoặc PNG.",
        )

    # 2. Đọc dữ liệu an toàn theo luồng (chunking 64 KB) để chống tràn bộ nhớ (OOM DoS)
    data = bytearray()
    chunk_size = 64 * 1024  # 64 KB
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Ảnh quá lớn (tối đa 8 MB).",
            )

    # 3. Đọc dữ liệu ảnh qua Pillow và kiểm tra định dạng thật của ảnh
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Không đọc được ảnh. Hãy thử ảnh khác.",
        )

    # Kiểm tra định dạng thực tế của ảnh (không tin phần mở rộng hay header client)
    if img.format not in ("JPEG", "PNG"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Chỉ chấp nhận ảnh JPG hoặc PNG.",
        )

    # 4. Đo thời gian xử lý và gọi mô hình
    t0 = time.perf_counter()
    result = await run_in_threadpool(classifier.predict, img)
    result["latency_ms"] = max(1, round((time.perf_counter() - t0) * 1000))

    return result
