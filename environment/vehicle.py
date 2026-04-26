from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Vehicle:
    car_id: int
    x: float
    y: float = 0.0
    velocity: float = 0.0
    path_type: str = "straight"  # "straight" or "merge"
    length: float = 4.5
    width: float = 1.8
    accel_pedal: float = 0.0
    brake_pedal: float = 0.0
    net_acceleration: float = 0.0
    last_net_acceleration: float = 0.0
    lane: int = 1  # 0=left, 1=center, 2=right (three-lane convention)
    vehicle_role: str = "passenger"  # "passenger", "ambulance"
    emergency_siren: bool = False
    last_lateral: str = "—"  # last discrete lane intent applied (broadcast / UI)

    def apply_action(
        self,
        accel_pedal: float,
        brake_pedal: float,
        dt: float,
        max_acceleration: float,
        max_deceleration: float,
        v_min: float,
        v_max: float,
    ) -> None:
        accel = float(np.clip(accel_pedal, 0.0, 1.0))
        brake = float(np.clip(brake_pedal, 0.0, 1.0))

        if accel > 0.0 and brake > 0.0:
            accel = 0.0

        self.accel_pedal = accel
        self.brake_pedal = brake

        self.last_net_acceleration = self.net_acceleration
        self.net_acceleration = (accel * max_acceleration) - (brake * max_deceleration)

        self.x = self.x + (self.velocity * dt) + (0.5 * self.net_acceleration * dt * dt)
        self.velocity = float(np.clip(self.velocity + self.net_acceleration * dt, v_min, v_max))

    def to_broadcast_packet(self) -> dict[str, Any]:
        return {
            "sender_id": int(self.car_id),
            "x_position": float(self.x),
            "y_position": float(self.y),
            "velocity": float(self.velocity),
            "path_type": str(self.path_type),
            "accel_pedal": float(self.accel_pedal),
            "brake_pedal": float(self.brake_pedal),
            "net_acceleration": float(self.net_acceleration),
            "length": float(self.length),
            "width": float(self.width),
            "lane_index": int(self.lane),
            "lateral_intent": str(self.last_lateral),
            "vehicle_role": str(self.vehicle_role),
            "emergency_siren": bool(self.emergency_siren),
        }
