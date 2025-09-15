# line_detector.py
import cv2
import numpy as np
from typing import List, Optional, Tuple

class RaceLineDetector:
    def __init__(self):
        self.calibration_frames = []
        self.track_template = None
        self.optimal_line = None

    def detect_track_boundaries(self, frame: np.ndarray) -> Tuple[Optional[List], Optional[List]]:
        """트랙 경계선 검출"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 전처리
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 적응적 임계값을 사용한 이진화
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 윤곽선 검출
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        # 가장 큰 윤곽선들을 트랙 경계로 가정
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        left_boundary = None
        right_boundary = None

        if len(contours) >= 2:
            # 왼쪽/오른쪽 구분 (x 좌표 기준)
            contour1_center = np.mean(contours[0][:, 0, 0])
            contour2_center = np.mean(contours[1][:, 0, 0])

            if contour1_center < contour2_center:
                left_boundary = contours[0]
                right_boundary = contours[1]
            else:
                left_boundary = contours[1]
                right_boundary = contours[0]

        return left_boundary, right_boundary

    def calculate_optimal_line(self, left_boundary: np.ndarray, right_boundary: np.ndarray) -> Optional[List]:
        """최적 레이스 라인 계산"""
        if left_boundary is None or right_boundary is None:
            return None

        # 경계선의 중점들을 연결하여 중앙선 생성
        optimal_points = []

        # Y 좌표 범위 설정
        min_y = max(np.min(left_boundary[:, 0, 1]), np.min(right_boundary[:, 0, 1]))
        max_y = min(np.max(left_boundary[:, 0, 1]), np.max(right_boundary[:, 0, 1]))

        # 일정 간격으로 샘플링
        for y in range(int(min_y), int(max_y), 10):
            # 해당 Y 좌표에서 왼쪽/오른쪽 경계의 X 좌표 찾기
            left_x = self.find_x_at_y(left_boundary, y)
            right_x = self.find_x_at_y(right_boundary, y)

            if left_x is not None and right_x is not None:
                # 중점 계산
                center_x = (left_x + right_x) // 2
                optimal_points.append([center_x, y])

        return optimal_points if optimal_points else None

    def find_x_at_y(self, contour: np.ndarray, target_y: int) -> Optional[int]:
        """특정 Y 좌표에서 윤곽선의 X 좌표 찾기"""
        points_at_y = contour[np.abs(contour[:, 0, 1] - target_y) < 5]

        if len(points_at_y) > 0:
            return int(np.mean(points_at_y[:, 0, 0]))

        return None

    def smooth_line(self, points: List[List[int]], window_size: int = 5) -> List[List[int]]:
        """레이스 라인 스무딩"""
        if len(points) < window_size:
            return points

        smoothed_points = []

        for i in range(len(points)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(points), i + window_size // 2 + 1)

            window_points = points[start_idx:end_idx]

            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)

            smoothed_points.append([int(avg_x), int(avg_y)])

        return smoothed_points

    def draw_race_line(self, frame: np.ndarray, optimal_line: List[List[int]],
                      color: Tuple[int, int, int] = (0, 255, 255), thickness: int = 3) -> np.ndarray:
        """레이스 라인을 프레임에 그리기"""
        if not optimal_line or len(optimal_line) < 2:
            return frame

        # 선 그리기
        for i in range(len(optimal_line) - 1):
            pt1 = tuple(optimal_line[i])
            pt2 = tuple(optimal_line[i + 1])
            cv2.line(frame, pt1, pt2, color, thickness)

        # 시작점과 끝점 표시
        if optimal_line:
            cv2.circle(frame, tuple(optimal_line[0]), 8, (0, 255, 0), -1)  # 시작점 (녹색)
            cv2.circle(frame, tuple(optimal_line[-1]), 8, (0, 0, 255), -1)  # 끝점 (빨간색)

        return frame

    def analyze_line_deviation(self, vehicle_center: Tuple[int, int],
                              optimal_line: List[List[int]]) -> float:
        """차량의 최적 라인 이탈 정도 분석"""
        if not optimal_line:
            return float('inf')

        # 가장 가까운 최적 라인 포인트 찾기
        min_distance = float('inf')

        for point in optimal_line:
            distance = np.sqrt((vehicle_center[0] - point[0])**2 + (vehicle_center[1] - point[1])**2)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def detect_racing_line_complete(self, frame: np.ndarray) -> dict:
        """완전한 레이스 라인 검출 및 분석"""
        # 트랙 경계 검출
        left_boundary, right_boundary = self.detect_track_boundaries(frame)

        # 최적 라인 계산
        optimal_line = None
        if left_boundary is not None and right_boundary is not None:
            optimal_line = self.calculate_optimal_line(left_boundary, right_boundary)
            if optimal_line:
                optimal_line = self.smooth_line(optimal_line)

        # 결과 프레임 생성
        result_frame = frame.copy()

        # 경계선 그리기
        if left_boundary is not None:
            cv2.drawContours(result_frame, [left_boundary], -1, (255, 0, 0), 2)
        if right_boundary is not None:
            cv2.drawContours(result_frame, [right_boundary], -1, (255, 0, 0), 2)

        # 최적 라인 그리기
        if optimal_line:
            result_frame = self.draw_race_line(result_frame, optimal_line)

        return {
            "frame": result_frame,
            "left_boundary": left_boundary,
            "right_boundary": right_boundary,
            "optimal_line": optimal_line,
            "has_track": left_boundary is not None and right_boundary is not None
        }