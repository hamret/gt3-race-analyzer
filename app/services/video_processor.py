import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time
from typing import List, Dict

class VideoProcessor:
    def __init__(self):
        # CUDA 사용 가능 확인
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # YOLOv8 모델 로드 (small과 medium 둘 다 준비)
        try:
            self.model_small = YOLO('yolov8s.pt').to(self.device)
            self.model_medium = YOLO('yolov8m.pt').to(self.device)
            self.current_model = self.model_small  # 기본은 small
            print("YOLOv8 models loaded successfully")
        except Exception as e:
            print(f"Model loading error: {e}")
            self.model_small = None
            self.model_medium = None
            self.current_model = None

    def switch_model(self, model_type="small"):
        """모델 전환 (small/medium)"""
        if model_type == "small" and self.model_small:
            self.current_model = self.model_small
        elif model_type == "medium" and self.model_medium:
            self.current_model = self.model_medium
        print(f"Switched to {model_type} model")

    def detect_vehicles(self, frame):
        """차량 검출"""
        if self.current_model is None:
            return []

        results = self.current_model(frame, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                # 차량 관련 클래스만 필터링 (자동차, 트럭, 버스 등)
                cls_id = int(box.cls.cpu().numpy()[0])
                if cls_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                    xyxy = box.xyxy.cpu().numpy()[0]
                    conf = float(box.conf.cpu().numpy()[0])

                    detection = {
                        "bbox": xyxy.tolist(),
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": result.names[cls_id]
                    }
                    detections.append(detection)

        return detections

    def detect_race_line(self, frame):
        """레이스 라인 검출"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny 엣지 검출
        edges = cv2.Canny(blurred, 50, 150)

        # 관심 영역 설정 (하단 부분)
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (0, height * 0.6),
            (width, height * 0.6),
            (width, height)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Hough 변환으로 직선 검출
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=50
        )

        return lines

    def draw_detections(self, frame, detections, lines=None):
        """검출 결과를 프레임에 그리기"""
        # 차량 박스 그리기
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            conf = det["confidence"]
            class_name = det["class_name"]

            # 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 라벨 그리기
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 레이스 라인 그리기
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        return frame

    def process_video(self, video_path):
        """비디오 전체 처리"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "비디오를 열 수 없습니다"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        results = {
            "total_frames": total_frames,
            "fps": fps,
            "total_vehicles": 0,
            "processing_fps": 0,
            "detections_per_frame": []
        }

        frame_count = 0
        start_time = time.time()

        # 샘플링 (모든 프레임 대신 일부만 처리)
        sample_rate = max(1, int(fps // 10))  # 초당 10프레임 정도만 처리

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 샘플링된 프레임만 처리
            if frame_count % sample_rate == 0:
                # 차량 검출
                detections = self.detect_vehicles(frame)

                # 레이스 라인 검출
                lines = self.detect_race_line(frame)

                frame_result = {
                    "frame_number": frame_count,
                    "vehicle_count": len(detections),
                    "detections": detections,
                    "has_race_line": lines is not None and len(lines) > 0
                }

                results["detections_per_frame"].append(frame_result)
                results["total_vehicles"] += len(detections)

        # 처리 완료
        end_time = time.time()
        processing_time = end_time - start_time
        results["processing_fps"] = (frame_count / sample_rate) / processing_time
        results["total_vehicles"] = results["total_vehicles"] // len(results["detections_per_frame"]) if results["detections_per_frame"] else 0

        cap.release()
        return results

    def process_frame_realtime(self, frame):
        """실시간 단일 프레임 처리"""
        start_time = time.time()

        # 차량 검출
        detections = self.detect_vehicles(frame)

        # 레이스 라인 검출
        lines = self.detect_race_line(frame)

        # 결과 그리기
        processed_frame = self.draw_detections(frame.copy(), detections, lines)

        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0

        return {
            "frame": processed_frame,
            "detections": detections,
            "lines": lines,
            "fps": fps,
            "vehicle_count": len(detections)
        }