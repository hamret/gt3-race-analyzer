import os
from typing import Optional

class Settings:
    # 프로젝트 설정
    PROJECT_NAME: str = "GT3 Race Analyzer"
    VERSION: str = "1.0.0"

    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # 파일 경로
    UPLOAD_DIR: str = "app/static/uploads"
    MODEL_DIR: str = "models"
    DATA_DIR: str = "data"

    # YOLO 설정
    YOLO_MODEL_SMALL: str = "yolov8s.pt"
    YOLO_MODEL_MEDIUM: str = "yolov8m.pt"
    YOLO_CONFIDENCE: float = 0.5
    YOLO_IOU_THRESHOLD: float = 0.45

    # 처리 설정
    MAX_VIDEO_SIZE: int = 100 * 1024 * 1024  # 100MB
    PROCESSING_FPS: int = 30
    FRAME_SKIP: int = 1  # 매 n번째 프레임만 처리

    # GPU 설정
    USE_CUDA: bool = True
    GPU_MEMORY_FRACTION: float = 0.8

    # 데이터베이스 설정 (향후 확장용)
    DATABASE_URL: Optional[str] = None

    def __init__(self):
        # 디렉토리 생성
        for directory in [self.UPLOAD_DIR, self.MODEL_DIR, self.DATA_DIR]:
            os.makedirs(directory, exist_ok=True)

settings = Settings()