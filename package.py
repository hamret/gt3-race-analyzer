import os

BASE = "gt3-race-analyzer"

folders = [
    f"{BASE}/app",
    f"{BASE}/app/api",
    f"{BASE}/app/models/yolo",
    f"{BASE}/app/models/custom",
    f"{BASE}/app/services",
    f"{BASE}/app/utils",
    f"{BASE}/app/static/css",
    f"{BASE}/app/static/js",
    f"{BASE}/app/static/uploads",
    f"{BASE}/frontend",
    f"{BASE}/models",
    f"{BASE}/data"
]

files = [
    f"{BASE}/app/__init__.py",
    f"{BASE}/app/main.py",
    f"{BASE}/app/api/__init__.py",
    f"{BASE}/app/api/video.py",
    f"{BASE}/app/api/analysis.py",
    f"{BASE}/app/api/websocket.py",
    f"{BASE}/app/models/__init__.py",
    f"{BASE}/app/services/__init__.py",
    f"{BASE}/app/services/video_processor.py",
    f"{BASE}/app/services/race_analyzer.py",
    f"{BASE}/app/services/line_detector.py",
    f"{BASE}/app/utils/__init__.py",
    f"{BASE}/app/utils/cv_utils.py",
    f"{BASE}/app/utils/db_utils.py",
    f"{BASE}/requirements.txt",
    f"{BASE}/config.py",
    f"{BASE}/README.md"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            f.write("# " + file.split("/")[-1] + "\n")
print("프로젝트 기본 구조 생성 완료!")
