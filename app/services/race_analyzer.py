import numpy as np
from typing import List, Dict, Tuple
import time

class RaceAnalyzer:
    def __init__(self):
        self.vehicle_tracks = {}  # 차량 추적 데이터
        self.lap_times = {}      # 랩타임 기록
        self.sector_times = {}   # 섹터별 타임
        self.positions = {}      # 현재 포지션
        self.track_sectors = []  # 트랙 섹터 정보

    def update_vehicle_position(self, vehicle_id: str, bbox: List[float], timestamp: float):
        """차량 위치 업데이트"""
        if vehicle_id not in self.vehicle_tracks:
            self.vehicle_tracks[vehicle_id] = []

        # 중심점 계산
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        self.vehicle_tracks[vehicle_id].append({
            "timestamp": timestamp,
            "center": (center_x, center_y),
            "bbox": bbox
        })

        # 최근 100개 포인트만 유지 (메모리 관리)
        if len(self.vehicle_tracks[vehicle_id]) > 100:
            self.vehicle_tracks[vehicle_id] = self.vehicle_tracks[vehicle_id][-100:]

    def calculate_speed(self, vehicle_id: str) -> float:
        """차량 속도 계산 (픽셀/초 기준)"""
        if vehicle_id not in self.vehicle_tracks or len(self.vehicle_tracks[vehicle_id]) < 2:
            return 0.0

        tracks = self.vehicle_tracks[vehicle_id]
        recent_tracks = tracks[-10:]  # 최근 10개 포인트 사용

        if len(recent_tracks) < 2:
            return 0.0

        # 거리와 시간 계산
        total_distance = 0
        total_time = 0

        for i in range(1, len(recent_tracks)):
            prev_pos = recent_tracks[i-1]["center"]
            curr_pos = recent_tracks[i]["center"]

            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            time_diff = recent_tracks[i]["timestamp"] - recent_tracks[i-1]["timestamp"]

            total_distance += distance
            total_time += time_diff

        return total_distance / total_time if total_time > 0 else 0.0

    def predict_overtaking_probability(self, vehicle1_id: str, vehicle2_id: str) -> float:
        """추월 가능성 예측"""
        if vehicle1_id not in self.vehicle_tracks or vehicle2_id not in self.vehicle_tracks:
            return 0.0

        # 두 차량의 최근 속도 비교
        speed1 = self.calculate_speed(vehicle1_id)
        speed2 = self.calculate_speed(vehicle2_id)

        # 두 차량의 현재 위치 비교
        if not self.vehicle_tracks[vehicle1_id] or not self.vehicle_tracks[vehicle2_id]:
            return 0.0

        pos1 = self.vehicle_tracks[vehicle1_id][-1]["center"]
        pos2 = self.vehicle_tracks[vehicle2_id][-1]["center"]

        # 거리 계산
        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

        # 속도 차이와 거리 기반 추월 확률 계산
        speed_advantage = speed1 - speed2

        if speed_advantage <= 0 or distance > 200:  # 너무 멀거나 느리면 추월 불가능
            return 0.0

        # 간단한 추월 확률 공식
        probability = min(0.95, max(0.05, speed_advantage / 50.0 * (200 - distance) / 200))

        return probability

    def analyze_race_line_efficiency(self, vehicle_id: str, race_lines: List) -> float:
        """레이스 라인 효율성 분석"""
        if vehicle_id not in self.vehicle_tracks or not race_lines:
            return 0.0

        # 차량 경로와 이상적 레이스 라인 비교
        recent_path = self.vehicle_tracks[vehicle_id][-20:]  # 최근 20개 포인트

        if len(recent_path) < 5:
            return 0.0

        # 간단한 효율성 계산 (실제로는 더 복잡한 알고리즘 필요)
        path_smoothness = self.calculate_path_smoothness(recent_path)

        return min(1.0, max(0.0, path_smoothness))

    def calculate_path_smoothness(self, path: List[Dict]) -> float:
        """경로의 부드러움 계산"""
        if len(path) < 3:
            return 0.0

        direction_changes = 0
        total_segments = len(path) - 2

        for i in range(1, len(path) - 1):
            prev_pos = path[i-1]["center"]
            curr_pos = path[i]["center"]
            next_pos = path[i+1]["center"]

            # 방향 벡터 계산
            vec1 = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
            vec2 = (next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])

            # 각도 변화 계산
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_change = np.arccos(cos_angle)

                if angle_change > np.pi / 4:  # 45도 이상 변화
                    direction_changes += 1

        smoothness = 1.0 - (direction_changes / total_segments)
        return smoothness

    def get_live_rankings(self) -> List[Dict]:
        """실시간 순위 계산"""
        rankings = []

        for vehicle_id, tracks in self.vehicle_tracks.items():
            if not tracks:
                continue

            latest_position = tracks[-1]["center"]
            speed = self.calculate_speed(vehicle_id)

            rankings.append({
                "vehicle_id": vehicle_id,
                "position": latest_position,
                "speed": speed,
                "track_length": len(tracks)
            })

        # Y 좌표 기준으로 순위 매기기 (위쪽이 앞선 것으로 가정)
        rankings.sort(key=lambda x: x["position"][1])

        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1

        return rankings

    def generate_race_summary(self) -> Dict:
        """레이스 요약 생성"""
        rankings = self.get_live_rankings()

        summary = {
            "total_vehicles": len(self.vehicle_tracks),
            "rankings": rankings,
            "average_speed": 0,
            "overtaking_predictions": []
        }

        # 평균 속도 계산
        if rankings:
            total_speed = sum(r["speed"] for r in rankings)
            summary["average_speed"] = total_speed / len(rankings)

        # 추월 예측 (상위 차량들 간)
        for i in range(min(5, len(rankings) - 1)):
            for j in range(i + 1, min(5, len(rankings))):
                vehicle1 = rankings[i]["vehicle_id"]
                vehicle2 = rankings[j]["vehicle_id"]
                probability = self.predict_overtaking_probability(vehicle2, vehicle1)

                if probability > 0.3:  # 30% 이상 확률만 포함
                    summary["overtaking_predictions"].append({
                        "overtaking_vehicle": vehicle2,
                        "target_vehicle": vehicle1,
                        "probability": probability
                    })

        return summary