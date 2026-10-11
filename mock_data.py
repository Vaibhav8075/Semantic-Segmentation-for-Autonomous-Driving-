"""
mock_data.py — Synthetic Road-Scene Generator
=============================================

Generates (image, mask) pairs that look like simplified driving scenes, for
developing and testing the pipeline without Cityscapes.

The image is *rendered* (asphalt texture, painted lane lines, sky, buildings,
cars with windows and wheels, people, signs on poles) rather than painted in
the label colors, so the model has to learn shapes and context instead of a
color lookup. Lane lines are painted on the road but labeled as Road, the
same way Cityscapes labels them.

Used by both prepare_dataset.py (writes samples to disk) and
dataset.MockSegDataset (generates samples on the fly).

Author: Vibhu
Project: Production-Grade Semantic Segmentation for Autonomous Driving
"""

from typing import Tuple

import cv2
import numpy as np

from config import BACKGROUND_CLASS, CLASS_NAMES

ROAD = CLASS_NAMES.index("Road")
VEHICLE = CLASS_NAMES.index("Vehicle")
PEDESTRIAN = CLASS_NAMES.index("Pedestrian")
SIDEWALK = CLASS_NAMES.index("Sidewalk")
TRAFFIC_SIGN = CLASS_NAMES.index("Traffic Sign")
BACKGROUND = BACKGROUND_CLASS


def _color(rng: np.random.Generator, base: Tuple[int, int, int], jitter: int) -> Tuple[int, int, int]:
    """Random color around a base RGB value."""
    c = np.array(base) + rng.integers(-jitter, jitter + 1, size=3)
    return tuple(int(v) for v in np.clip(c, 0, 255))


def _road_edges(y: float, horizon: int, h: int, vp_x: float, w: int,
                top_half: float, bottom_half: float) -> Tuple[float, float]:
    """Left/right x of the road at row y (linear perspective)."""
    t = (y - horizon) / max(1, h - horizon)
    half = top_half + (bottom_half - top_half) * t
    return vp_x - half * w, vp_x + half * w


def generate_mock_sample(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render one synthetic road scene.

    Args:
        height: Image height.
        width: Image width.
        rng: NumPy random generator (seed it for reproducible samples).

    Returns:
        image: RGB image (H, W, 3), uint8.
        mask: Class-index mask (H, W), uint8, values in [0, NUM_CLASSES).
    """
    h, w = height, width
    image = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.full((h, w), BACKGROUND, dtype=np.uint8)

    horizon = int(h * rng.uniform(0.35, 0.5))
    vp_x = w * rng.uniform(0.4, 0.6)  # vanishing point x

    # ---- Sky (vertical gradient) ----
    top = np.array(_color(rng, (110, 160, 220), 30), dtype=np.float32)
    bottom = np.array(_color(rng, (200, 215, 230), 20), dtype=np.float32)
    t = np.linspace(0, 1, horizon, dtype=np.float32)[:, None]
    image[:horizon] = (top * (1 - t) + bottom * t).astype(np.uint8)[:, None, :]

    # ---- Ground outside the road (terrain / grass) ----
    image[horizon:] = _color(rng, (95, 110, 70), 25)

    # ---- Buildings and trees along the horizon ----
    x = -int(rng.integers(0, 40))
    while x < w:
        bw = int(rng.integers(w // 20, w // 7))
        if rng.random() < 0.7:
            top_y = int(rng.integers(int(h * 0.05), max(int(h * 0.05) + 1, horizon - 10)))
            col = _color(rng, (150, 130, 115), 45)
            cv2.rectangle(image, (x, top_y), (x + bw, horizon + 2), col, -1)
            # Windows
            win = tuple(int(c * 0.55) for c in col)
            for wy in range(top_y + 8, horizon - 8, 18):
                for wx in range(x + 6, x + bw - 8, 14):
                    cv2.rectangle(image, (wx, wy), (wx + 6, wy + 8), win, -1)
        else:
            cx, r = x + bw // 2, max(6, bw // 2)
            cv2.rectangle(image, (cx - 3, horizon - r), (cx + 3, horizon + 2), (80, 60, 40), -1)
            cv2.circle(image, (cx, horizon - r), r, _color(rng, (60, 120, 50), 25), -1)
        x += bw + int(rng.integers(0, w // 30 + 1))

    # ---- Road (trapezoid to the vanishing point) ----
    top_half = rng.uniform(0.02, 0.05)
    bottom_half = rng.uniform(0.45, 0.7)
    rl_top, rr_top = _road_edges(horizon, horizon, h, vp_x, w, top_half, bottom_half)
    rl_bot, rr_bot = _road_edges(h, horizon, h, vp_x, w, top_half, bottom_half)
    road_poly = np.array([[rl_top, horizon], [rr_top, horizon],
                          [rr_bot, h], [rl_bot, h]], dtype=np.int32)
    asphalt = _color(rng, (90, 90, 95), 20)
    cv2.fillPoly(image, [road_poly], asphalt)
    cv2.fillPoly(mask, [road_poly], ROAD)

    # ---- Sidewalks (bands on both sides of the road) ----
    sw = rng.uniform(0.04, 0.12)  # sidewalk width at the bottom, as a fraction of w
    sw_top = sw * top_half / bottom_half
    walk = _color(rng, (175, 170, 160), 20)
    left_walk = np.array([[rl_top - sw_top * w, horizon], [rl_top, horizon],
                          [rl_bot, h], [rl_bot - sw * w, h]], dtype=np.int32)
    right_walk = np.array([[rr_top, horizon], [rr_top + sw_top * w, horizon],
                           [rr_bot + sw * w, h], [rr_bot, h]], dtype=np.int32)
    for poly in (left_walk, right_walk):
        cv2.fillPoly(image, [poly], walk)
        cv2.fillPoly(mask, [poly], SIDEWALK)

    # ---- Lane markings: painted on the image only (labeled as Road) ----
    paint = (235, 235, 230) if rng.random() < 0.7 else (230, 200, 60)
    n_lanes = int(rng.integers(2, 4))
    dash = rng.random() < 0.7
    for k in range(1, n_lanes):
        frac = k / n_lanes
        y = float(horizon)
        while y < h:
            seg = max(4.0, (y - horizon) * 0.25 + 4)
            y2 = min(float(h), y + seg)
            l1, r1 = _road_edges(y, horizon, h, vp_x, w, top_half, bottom_half)
            l2, r2 = _road_edges(y2, horizon, h, vp_x, w, top_half, bottom_half)
            x1, x2 = l1 + (r1 - l1) * frac, l2 + (r2 - l2) * frac
            thick = max(1, int(1 + 6 * (y2 - horizon) / (h - horizon)))
            cv2.line(image, (int(x1), int(y)), (int(x2), int(y2)), paint, thick)
            y = y2 + (seg * 0.8 if dash else 0)

    # ---- Vehicles (on the road, sized by perspective, far-to-near order) ----
    n_vehicles = int(rng.integers(0, 6))
    ys = np.sort(rng.uniform(horizon + 0.08 * (h - horizon), h * 0.98, n_vehicles))
    for y_bottom in ys:
        scale = (y_bottom - horizon) / (h - horizon)
        vw = max(8, int(w * 0.18 * scale + 6))
        vh = max(6, int(vw * rng.uniform(0.6, 0.85)))
        left, right = _road_edges(y_bottom, horizon, h, vp_x, w, top_half, bottom_half)
        if right - left <= vw:
            continue
        x0 = int(rng.uniform(left, right - vw))
        y1, y0 = int(y_bottom), int(y_bottom) - vh
        body = _color(rng, tuple(int(c) for c in rng.integers(30, 230, size=3)), 0)
        cv2.rectangle(image, (x0, y0 + vh // 3), (x0 + vw, y1), body, -1)
        cv2.rectangle(image, (x0 + vw // 6, y0), (x0 + vw - vw // 6, y0 + vh // 3 + 1), body, -1)
        cv2.rectangle(image, (x0 + vw // 5, y0 + vh // 12), (x0 + vw - vw // 5, y0 + vh // 3),
                      (40, 50, 60), -1)  # windows
        wheel = max(2, vw // 8)
        cv2.circle(image, (x0 + wheel + 1, y1 - 1), wheel, (20, 20, 20), -1)
        cv2.circle(image, (x0 + vw - wheel - 1, y1 - 1), wheel, (20, 20, 20), -1)
        cv2.rectangle(mask, (x0, y0 + vh // 3), (x0 + vw, y1), VEHICLE, -1)
        cv2.rectangle(mask, (x0 + vw // 6, y0), (x0 + vw - vw // 6, y0 + vh // 3 + 1), VEHICLE, -1)
        cv2.circle(mask, (x0 + wheel + 1, y1 - 1), wheel, VEHICLE, -1)
        cv2.circle(mask, (x0 + vw - wheel - 1, y1 - 1), wheel, VEHICLE, -1)

    # ---- Pedestrians (mostly on the sidewalks) ----
    for _ in range(int(rng.integers(0, 4))):
        y_feet = rng.uniform(horizon + 0.1 * (h - horizon), h * 0.97)
        scale = (y_feet - horizon) / (h - horizon)
        ph = max(10, int(h * 0.3 * scale + 8))
        pw = max(4, ph // 4)
        left, right = _road_edges(y_feet, horizon, h, vp_x, w, top_half, bottom_half)
        walk_w = sw * w * scale
        if rng.random() < 0.85:
            cx = left - rng.uniform(0.2, 0.8) * walk_w if rng.random() < 0.5 \
                else right + rng.uniform(0.2, 0.8) * walk_w
        else:
            cx = rng.uniform(left, right)
        cx, y1 = int(cx), int(y_feet)
        head = max(2, pw // 2)
        shirt = tuple(int(c) for c in rng.integers(20, 240, size=3))
        cv2.rectangle(image, (cx - pw // 2, y1 - ph + 2 * head), (cx + pw // 2, y1 - ph // 2), shirt, -1)
        cv2.rectangle(image, (cx - pw // 2, y1 - ph // 2), (cx + pw // 2, y1), _color(rng, (50, 50, 80), 30), -1)
        cv2.circle(image, (cx, y1 - ph + head), head, _color(rng, (200, 160, 130), 40), -1)
        cv2.rectangle(mask, (cx - pw // 2, y1 - ph + 2 * head), (cx + pw // 2, y1), PEDESTRIAN, -1)
        cv2.circle(mask, (cx, y1 - ph + head), head, PEDESTRIAN, -1)

    # ---- Traffic signs on poles (pole is Background, plate is Traffic Sign) ----
    for _ in range(int(rng.integers(0, 3))):
        y_base = rng.uniform(horizon + 0.05 * (h - horizon), horizon + 0.6 * (h - horizon))
        scale = (y_base - horizon) / (h - horizon)
        left, right = _road_edges(y_base, horizon, h, vp_x, w, top_half, bottom_half)
        side = rng.random() < 0.5
        px = int(left - sw * w * scale * 0.9) if side else int(right + sw * w * scale * 0.9)
        pole_h = int(h * 0.35 * scale + 15)
        size = max(5, int(h * 0.07 * scale + 5))
        px = int(np.clip(px, size, w - size - 1))
        y_top = int(y_base) - pole_h
        cv2.rectangle(image, (px - 1, y_top), (px + 1, int(y_base)), (110, 110, 110), -1)
        plate = [(200, 30, 30), (30, 70, 180), (230, 190, 30)][int(rng.integers(0, 3))]
        if rng.random() < 0.5:
            cv2.circle(image, (px, y_top), size, (240, 240, 240), -1)
            cv2.circle(image, (px, y_top), max(2, size - 2), plate, -1)
            cv2.circle(mask, (px, y_top), size, TRAFFIC_SIGN, -1)
        else:
            cv2.rectangle(image, (px - size, y_top - size), (px + size, y_top + size), (240, 240, 240), -1)
            cv2.rectangle(image, (px - size + 2, y_top - size + 2), (px + size - 2, y_top + size - 2), plate, -1)
            cv2.rectangle(mask, (px - size, y_top - size), (px + size, y_top + size), TRAFFIC_SIGN, -1)

    # ---- Photometric variation: lighting, sensor noise, slight blur ----
    img = image.astype(np.float32)
    img = img * rng.uniform(0.7, 1.25) + rng.uniform(-20, 20)
    img += rng.normal(0, rng.uniform(3, 10), img.shape)
    image = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    return image, mask
