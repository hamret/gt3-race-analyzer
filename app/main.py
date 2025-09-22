from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import cv2
import json
from app.services.video_processor import VideoProcessor
from app.services.race_analyzer import RaceAnalyzer
import asyncio

app = FastAPI(title="GT3 Race Analyzer", version="1.0.0")

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 템플릿 설정
templates = Jinja2Templates(directory="app/templates")

# 업로드 디렉토리 생성
UPLOAD_DIR = "app/static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 전역 인스턴스
video_processor = VideoProcessor()
race_analyzer = RaceAnalyzer()

@app.get("/broadcast", response_class=HTMLResponse)
async def get_broadcast_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/")
async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GT3 Race Analyzer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            button { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #005a87; }
            #video { max-width: 100%; }
            .stats { display: flex; gap: 20px; margin: 20px 0; }
            .stat-box { background: #f5f5f5; padding: 15px; border-radius: 8px; flex: 1; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏁 GT3 Race Analyzer</h1>

            <div class="upload-area">
                <input type="file" id="videoFile" accept="video/*" style="display: none;">
                <button onclick="document.getElementById('videoFile').click()">비디오 업로드</button>
                <p>GT3 레이스 영상을 업로드하세요</p>
            </div>

            <video id="video" controls style="display: none;"></video>

            <div class="stats">
                <div class="stat-box">
                    <h3>검출된 차량</h3>
                    <div id="vehicleCount">0</div>
                </div>
                <div class="stat-box">
                    <h3>재생 FPS</h3>
                    <div id="fps">0</div>
                </div>
                <div class="stat-box">
                    <h3>분석 상태</h3>
                    <div id="status">대기중</div>
                </div>
            </div>

            <div id="results"></div>
        </div>

        <script>
            const videoFile = document.getElementById('videoFile');
            const video = document.getElementById('video');
            const resultsDiv = document.getElementById('results');

            videoFile.addEventListener('change', async (e) => {
                const file = e.target.files[0];
                if (file) {
                    const formData = new FormData();
                    formData.append('file', file);

                    try {
                        const response = await fetch('/upload-video', {
                            method: 'POST',
                            body: formData
                        });
                        const result = await response.json();

                        video.src = '/static/uploads/' + result.filename;
                        video.style.display = 'block';
                        document.getElementById('status').textContent = '업로드 완료';

                        // 분석 시작
                        startAnalysis(result.filename);
                    } catch (error) {
                        console.error('업로드 실패:', error);
                    }
                }
            });

            function startAnalysis(filename) {
                fetch('/analyze-video/' + filename, { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').textContent = '분석 완료';
                        document.getElementById('vehicleCount').textContent = data.total_vehicles;
                        document.getElementById('fps').textContent = data.fps.toFixed(1);

                        resultsDiv.innerHTML = '<h3>분석 결과</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    });
            }
        </script>
    </body>
    </html>
    """)

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"filename": file.filename, "path": filepath, "size": len(content)}

@app.post("/analyze-video/{filename}")
async def analyze_video(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        return {"error": "파일을 찾을 수 없습니다"}

    results = video_processor.process_video(filepath)
    return results

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message["type"] == "frame":
                # 실시간 프레임 처리
                result = {"type": "result", "data": "processed"}
                await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)