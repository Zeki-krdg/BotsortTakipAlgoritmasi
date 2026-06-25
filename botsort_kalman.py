"""
BOT-SORT Kalman Filtresi — Kamikaze İHA Takip Modülü
=====================================================

8 boyutlu Kalman filtresi: [cx, cy, w, h, vx, vy, vw, vh]
"""

import numpy as np


def iou(bbox1, bbox2):
    """İki [x1,y1,x2,y2] bbox arası IoU hesapla."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


class KalmanBoxTracker:
    """8 boyutlu sabit-hız Kalman filtresi ile bbox takibi."""

    def __init__(self, bbox_xyxy):
        """bbox_xyxy: [x1, y1, x2, y2]"""
        self.dim_x = 8
        self.dim_z = 4

        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        # Durum: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.zeros((self.dim_x, 1), dtype=np.float64)
        self.x[0, 0] = cx
        self.x[1, 0] = cy
        self.x[2, 0] = w
        self.x[3, 0] = h

        # F: Durum geçişi (sabit hız modeli)
        self.F = np.eye(self.dim_x, dtype=np.float64)
        self.F[0, 4] = 1.0  # cx += vx
        self.F[1, 5] = 1.0  # cy += vy
        # Kutu boyutu (w, h) tahmini KAPATILDI! 
        # Böylece kutu boyutu kendi kendine büyüyüp küçülmeyecek, sadece YOLO'dan gelen boyutta kalacak.
        # self.F[2, 6] = 1.0  
        # self.F[3, 7] = 1.0  

        # H: Ölçüm matrisi
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # P: Başlangıç hata kovaryansı
        self.P = np.diag([10, 10, 10, 10, 100, 100, 25, 25]).astype(np.float64)

        # Q: Proses gürültüsü
        self.Q = np.diag([1, 1, 0.5, 0.5, 16, 16, 0.0, 0.0]).astype(np.float64)

        # R: Ölçüm gürültüsü — YOLO tespiti ÇOK GÜVENİLİR
        # w ve h için 0.01 veriyoruz ki Kalman eski boyutunu unutup direkt YOLO'nun verdiği boyuta %100 kilitlensin.
        self.R = np.diag([0.5, 0.5, 0.01, 0.01]).astype(np.float64)

        self.time_since_update = 0
        self.hits = 0

    def predict(self):
        """Bir sonraki kareyi tahmin et. [x1,y1,x2,y2] döndürür."""
        if self.x[2, 0] + self.x[6, 0] <= 0:
            self.x[6, 0] = 0.0
        if self.x[3, 0] + self.x[7, 0] <= 0:
            self.x[7, 0] = 0.0

        # Hız Sönümleme (Velocity Damping) - SADECE GERÇEKTEN KAYBOLDUYSA
        # YOLO'nun çalışmadığı normal ara karelerde fren YAPMA!
        # Sadece 3 kareden uzun süre hedef bulunamazsa (drone gerçekten kayıpsa) fren yap
        if self.time_since_update > 3:
            self.x[4, 0] *= 0.85  # vx
            self.x[5, 0] *= 0.85  # vy
            self.x[6, 0] *= 0.50  # vw
            self.x[7, 0] *= 0.50  # vh

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self._get_bbox()

    def update(self, bbox_xyxy):
        """YOLO tespiti ile Kalman'ı güncelle."""
        x1, y1, x2, y2 = bbox_xyxy
        z = np.array([[(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]], dtype=np.float64).T

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1

    def _get_bbox(self):
        """[x1, y1, x2, y2] döndür."""
        cx, cy = self.x[0, 0], self.x[1, 0]
        w = max(4.0, self.x[2, 0])
        h = max(4.0, self.x[3, 0])
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    def get_bbox(self):
        return self._get_bbox()

    def get_confidence(self):
        if self.time_since_update == 0:
            return 1.0
        return max(0.0, 0.96 ** self.time_since_update)
