import math
from dataclasses import dataclass

Point = tuple[int, int]

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def hand_scale(points: list[Point]) -> float:
    return distance(points[WRIST], points[MIDDLE_MCP])

def pinch_ratio(points: list[Point]) -> float:
    scale = hand_scale(points)
    if scale == 0:
        return 1.0
    return distance(points[THUMB_TIP], points[INDEX_TIP]) / scale

def is_pinching(points: list[Point], threshold: float = 0.25) -> bool:
    return pinch_ratio(points) < threshold

def correct_side(side: str) -> str:
    # Frame is flipped due to cv2.flip(frame, 1), so it was needed to correct the side.
    return "Right" if side == "Left" else "Left"

@dataclass
class HandState:
    side: str
    points: list[Point]
    pinching: bool
    thumb_tip: Point
    index_tip: Point

def read_hand(points: list[Point], raw_side: str) -> HandState:
    return HandState(
        side=correct_side(raw_side),
        points=points,
        pinching=is_pinching(points),
        thumb_tip=points[THUMB_TIP],
        index_tip=points[INDEX_TIP],
    )