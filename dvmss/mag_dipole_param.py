from dataclasses import dataclass


@dataclass
class MagDipoleParam:
    position: tuple[float, float, float]
    orientation: tuple[float, float]
    moment: float
