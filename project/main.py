import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import threading
import time
import os
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# LOCAL MODEL
# Dat hand_landmarker.task cung thu muc voi main.py
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

print("Model path:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Khong tim thay model: {MODEL_PATH}\n"
        "Hay dat hand_landmarker.task cung thu muc voi main.py"
    )

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

hand_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    hand_options
)


# ============================================================
# SETTINGS
# ============================================================

WIDTH = 1920
HEIGHT = 1080
NUM_PARTICLES = 2500


# ============================================================
# SHARED DATA
# ============================================================

lock = threading.Lock()

shared_data = {
    "mode": 1,
    "finger_count": 0,
    "target_x": 0.0,
    "target_y": 0.0,
    "target_z": -12.0,
    "frame": None,
    "running": True
}


# ============================================================
# IMAGE -> PARTICLES
# ============================================================

def image_to_particles(img, num_particles, scale=5.0):
    mask = np.any(img > 0, axis=2)
    y_indices, x_indices = np.where(mask)

    if len(x_indices) == 0:
        raise ValueError("Shape image khong co pixel nao.")

    selected = np.random.choice(
        len(x_indices),
        num_particles,
        replace=True
    )

    xs = x_indices[selected]
    ys = y_indices[selected]

    h, w, _ = img.shape

    x = ((xs - w / 2) / w) * scale
    y = -((ys - h / 2) / h) * scale
    z = np.random.uniform(-0.05, 0.05, num_particles)

    positions = np.stack([x, y, z], axis=1)

    bgr = img[ys, xs].astype(np.float64) / 255.0
    colors = bgr[:, ::-1]

    return positions, colors


# ============================================================
# AVENGERS
# 0 FINGERS / DEFAULT
# ============================================================

def create_avengers():
    size = 760

    img = np.zeros(
        (size, size, 3),
        dtype=np.uint8
    )

    center = (
        size // 2,
        size // 2
    )

    red = (20, 20, 245)
    dark_red = (10, 10, 150)

    cv2.circle(
        img,
        center,
        292,
        dark_red,
        60
    )

    cv2.circle(
        img,
        center,
        270,
        red,
        32
    )

    a_body = np.array([
        [165, 675],
        [330, 180],
        [410, 180],
        [575, 675],
        [485, 675],
        [445, 555],
        [300, 555],
        [255, 675]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [a_body],
        red
    )

    hole = np.array([
        [362, 350],
        [405, 500],
        [315, 500]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [hole],
        (0, 0, 0)
    )

    arrow = np.array([
        [260, 500],
        [455, 500],
        [520, 445],
        [520, 535],
        [450, 590],
        [245, 590]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [arrow],
        red
    )

    return image_to_particles(
        img,
        NUM_PARTICLES,
        scale=5.8
    )


# ============================================================
# IRON MAN
# 1 FINGER
# ============================================================

def create_ironman():
    h = 760
    w = 600

    img = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    red = (25, 35, 230)
    red_dark = (15, 20, 150)
    gold = (35, 180, 235)
    gold_light = (80, 220, 255)
    dark = (15, 15, 15)
    white = (255, 255, 255)

    helmet = np.array([
        [300, 30],
        [190, 55],
        [105, 130],
        [70, 250],
        [85, 500],
        [145, 650],
        [230, 720],
        [300, 742],
        [370, 720],
        [455, 650],
        [515, 500],
        [530, 250],
        [495, 130],
        [410, 55]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [helmet],
        red
    )

    face = np.array([
        [160, 175],
        [225, 135],
        [250, 265],
        [350, 265],
        [375, 135],
        [440, 175],
        [460, 305],
        [445, 505],
        [390, 630],
        [345, 590],
        [255, 590],
        [210, 630],
        [155, 505],
        [140, 305]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [face],
        gold
    )

    forehead = np.array([
        [155, 100],
        [300, 42],
        [445, 100],
        [375, 140],
        [350, 265],
        [250, 265],
        [225, 140]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [forehead],
        red
    )

    left_cheek = np.array([
        [145, 305],
        [185, 355],
        [210, 500],
        [160, 505]
    ], dtype=np.int32)

    right_cheek = np.array([
        [455, 305],
        [415, 355],
        [390, 500],
        [440, 505]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [left_cheek],
        gold_light
    )

    cv2.fillPoly(
        img,
        [right_cheek],
        gold_light
    )

    jaw = np.array([
        [210, 630],
        [255, 590],
        [345, 590],
        [390, 630],
        [360, 690],
        [240, 690]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [jaw],
        red_dark
    )

    left_eye_border = np.array([
        [160, 300],
        [275, 315],
        [250, 360],
        [175, 348]
    ], dtype=np.int32)

    right_eye_border = np.array([
        [325, 315],
        [440, 300],
        [425, 348],
        [350, 360]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [left_eye_border],
        dark
    )

    cv2.fillPoly(
        img,
        [right_eye_border],
        dark
    )

    left_eye = np.array([
        [178, 312],
        [260, 323],
        [242, 343],
        [190, 336]
    ], dtype=np.int32)

    right_eye = np.array([
        [340, 323],
        [422, 312],
        [410, 336],
        [358, 343]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [left_eye],
        white
    )

    cv2.fillPoly(
        img,
        [right_eye],
        white
    )

    cv2.line(
        img,
        (160, 175),
        (140, 305),
        dark,
        6
    )

    cv2.line(
        img,
        (440, 175),
        (460, 305),
        dark,
        6
    )

    cv2.line(
        img,
        (155, 505),
        (210, 630),
        dark,
        6
    )

    cv2.line(
        img,
        (445, 505),
        (390, 630),
        dark,
        6
    )

    cv2.line(
        img,
        (255, 590),
        (345, 590),
        dark,
        6
    )

    return image_to_particles(
        img,
        NUM_PARTICLES,
        scale=5.7
    )


# ============================================================
# CAPTAIN AMERICA SHIELD
# 2 FINGERS
# ============================================================

def create_captain_shield():
    size = 700

    img = np.zeros(
        (size, size, 3),
        dtype=np.uint8
    )

    center = (
        size // 2,
        size // 2
    )

    cv2.circle(
        img,
        center,
        300,
        (20, 20, 225),
        -1
    )

    cv2.circle(
        img,
        center,
        235,
        (245, 245, 245),
        -1
    )

    cv2.circle(
        img,
        center,
        175,
        (20, 20, 220),
        -1
    )

    cv2.circle(
        img,
        center,
        120,
        (155, 65, 20),
        -1
    )

    points = []

    cx, cy = center
    outer_r = 90
    inner_r = 38

    for i in range(10):
        angle = (
            -math.pi / 2
            + i * math.pi / 5
        )

        radius = (
            outer_r
            if i % 2 == 0
            else inner_r
        )

        px = int(
            cx
            + radius * math.cos(angle)
        )

        py = int(
            cy
            + radius * math.sin(angle)
        )

        points.append([
            px,
            py
        ])

    points = np.array(
        points,
        dtype=np.int32
    )

    cv2.fillPoly(
        img,
        [points],
        (255, 255, 255)
    )

    return image_to_particles(
        img,
        NUM_PARTICLES,
        scale=5.6
    )


# ============================================================
# SPIDER-MAN
# 3 FINGERS
# ============================================================

def create_spiderman():
    h = 760
    w = 600

    img = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    red = (25, 35, 235)
    dark = (12, 12, 12)
    white = (245, 245, 245)

    head = np.array([
        [300, 35],
        [195, 55],
        [115, 125],
        [75, 250],
        [90, 450],
        [145, 610],
        [235, 720],
        [300, 750],
        [365, 720],
        [455, 610],
        [510, 450],
        [525, 250],
        [485, 125],
        [405, 55]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [head],
        red
    )

    center = (
        300,
        385
    )

    web_targets = [
        (300, 45),
        (190, 70),
        (100, 180),
        (80, 380),
        (135, 590),
        (235, 720),
        (365, 720),
        (465, 590),
        (520, 380),
        (500, 180),
        (410, 70)
    ]

    for target in web_targets:
        cv2.line(
            img,
            center,
            target,
            dark,
            6
        )

    cv2.ellipse(
        img,
        center,
        (75, 70),
        0,
        0,
        360,
        dark,
        5
    )

    cv2.ellipse(
        img,
        center,
        (145, 135),
        0,
        0,
        360,
        dark,
        5
    )

    cv2.ellipse(
        img,
        center,
        (220, 215),
        0,
        0,
        360,
        dark,
        5
    )

    left_eye_outer = np.array([
        [115, 285],
        [248, 335],
        [232, 405],
        [155, 490],
        [98, 410]
    ], dtype=np.int32)

    right_eye_outer = np.array([
        [352, 335],
        [485, 285],
        [502, 410],
        [445, 490],
        [368, 405]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [left_eye_outer],
        dark
    )

    cv2.fillPoly(
        img,
        [right_eye_outer],
        dark
    )

    left_eye = np.array([
        [132, 308],
        [229, 345],
        [215, 392],
        [160, 457],
        [120, 400]
    ], dtype=np.int32)

    right_eye = np.array([
        [371, 345],
        [468, 308],
        [480, 400],
        [440, 457],
        [385, 392]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [left_eye],
        white
    )

    cv2.fillPoly(
        img,
        [right_eye],
        white
    )

    return image_to_particles(
        img,
        NUM_PARTICLES,
        scale=5.7
    )


# ============================================================
# THOR MJOLNIR
# 4 FINGERS
# ============================================================

def create_thor_hammer():
    h = 760
    w = 760

    img = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    dark = (30, 30, 35)
    metal = (150, 165, 175)
    metal_light = (220, 225, 235)
    blue = (255, 180, 30)

    hammer = np.array([
        [225, 235],
        [470, 190],
        [535, 255],
        [510, 395],
        [275, 430],
        [215, 355]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [hammer],
        metal
    )

    highlight = np.array([
        [250, 245],
        [460, 210],
        [500, 250],
        [285, 290]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [highlight],
        metal_light
    )

    bottom = np.array([
        [275, 390],
        [510, 355],
        [510, 395],
        [275, 430]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [bottom],
        dark
    )

    cv2.line(
        img,
        (305, 415),
        (195, 645),
        dark,
        34
    )

    cv2.line(
        img,
        (306, 420),
        (200, 638),
        (80, 85, 90),
        23
    )

    for i in range(7):
        x = int(
            296 - i * 15
        )

        y = int(
            440 + i * 31
        )

        cv2.line(
            img,
            (x - 12, y - 6),
            (x + 14, y + 8),
            metal_light,
            4
        )

    cv2.circle(
        img,
        (190, 655),
        20,
        metal,
        -1
    )

    bolts = [
        [
            (330, 205),
            (310, 145),
            (350, 105),
            (325, 45)
        ],
        [
            (415, 195),
            (445, 135),
            (425, 95),
            (470, 35)
        ],
        [
            (240, 270),
            (185, 215),
            (215, 160),
            (160, 105)
        ],
        [
            (515, 260),
            (575, 210),
            (545, 155),
            (605, 100)
        ],
        [
            (430, 395),
            (470, 460),
            (445, 505),
            (490, 565)
        ]
    ]

    for bolt in bolts:
        for i in range(
            len(bolt) - 1
        ):
            cv2.line(
                img,
                bolt[i],
                bolt[i + 1],
                blue,
                8
            )

    cv2.line(
        img,
        (310, 145),
        (275, 120),
        blue,
        4
    )

    cv2.line(
        img,
        (445, 135),
        (485, 115),
        blue,
        4
    )

    cv2.line(
        img,
        (185, 215),
        (145, 230),
        blue,
        4
    )

    cv2.line(
        img,
        (575, 210),
        (620, 225),
        blue,
        4
    )

    return image_to_particles(
        img,
        NUM_PARTICLES,
        scale=6.0
    )


# ============================================================
# HULK
# 5 FINGERS
# ============================================================

def create_hulk():
    h = 760
    w = 620

    img = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    green = (70, 180, 75)
    green_light = (90, 210, 100)
    green_dark = (35, 120, 45)
    dark = (20, 25, 20)
    white = (245, 245, 245)

    face = np.array([
        [110, 170],
        [165, 95],
        [250, 70],
        [370, 72],
        [460, 105],
        [515, 175],
        [540, 300],
        [510, 470],
        [445, 600],
        [350, 680],
        [275, 680],
        [180, 620],
        [110, 520],
        [80, 380]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [face],
        green
    )

    hair = np.array([
        [100, 190],
        [125, 110],
        [165, 55],
        [210, 90],
        [250, 40],
        [300, 75],
        [350, 35],
        [390, 75],
        [445, 50],
        [485, 115],
        [520, 180],
        [470, 150],
        [400, 130],
        [330, 125],
        [250, 130],
        [180, 150]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [hair],
        dark
    )

    cv2.line(
        img,
        (250, 175),
        (290, 235),
        green_dark,
        8
    )

    cv2.line(
        img,
        (370, 175),
        (330, 235),
        green_dark,
        8
    )

    left_brow = np.array([
        [145, 260],
        [255, 225],
        [285, 255],
        [175, 290]
    ], dtype=np.int32)

    right_brow = np.array([
        [335, 255],
        [365, 225],
        [475, 260],
        [445, 290]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [left_brow],
        dark
    )

    cv2.fillPoly(
        img,
        [right_brow],
        dark
    )

    cv2.ellipse(
        img,
        (220, 320),
        (48, 20),
        -12,
        0,
        360,
        white,
        -1
    )

    cv2.ellipse(
        img,
        (400, 320),
        (48, 20),
        12,
        0,
        360,
        white,
        -1
    )

    cv2.circle(
        img,
        (225, 320),
        10,
        dark,
        -1
    )

    cv2.circle(
        img,
        (395, 320),
        10,
        dark,
        -1
    )

    nose = np.array([
        [292, 345],
        [260, 430],
        [310, 470],
        [360, 430],
        [328, 345]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [nose],
        green_dark
    )

    cv2.line(
        img,
        (310, 360),
        (310, 435),
        green_light,
        5
    )

    mouth_outer = np.array([
        [165, 525],
        [235, 490],
        [385, 490],
        [455, 525],
        [405, 595],
        [215, 595]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [mouth_outer],
        dark
    )

    teeth = np.array([
        [195, 520],
        [425, 520],
        [395, 558],
        [225, 558]
    ], dtype=np.int32)

    cv2.fillPoly(
        img,
        [teeth],
        white
    )

    for x in [
        250,
        290,
        330,
        370
    ]:
        cv2.line(
            img,
            (x, 520),
            (x, 558),
            (100, 100, 100),
            2
        )

    cv2.line(
        img,
        (145, 390),
        (195, 420),
        green_dark,
        6
    )

    cv2.line(
        img,
        (475, 390),
        (425, 420),
        green_dark,
        6
    )

    return image_to_particles(
        img,
        NUM_PARTICLES,
        scale=5.8
    )


# ============================================================
# CREATE ALL SHAPES
# ============================================================

print("Dang tao AVENGERS...")
pos_avengers, color_avengers = create_avengers()

print("Dang tao IRON MAN...")
pos_ironman, color_ironman = create_ironman()

print("Dang tao CAPTAIN AMERICA...")
pos_captain, color_captain = create_captain_shield()

print("Dang tao SPIDER-MAN...")
pos_spiderman, color_spiderman = create_spiderman()

print("Dang tao THOR...")
pos_thor, color_thor = create_thor_hammer()

print("Dang tao HULK...")
pos_hulk, color_hulk = create_hulk()

print("Da tao xong tat ca particle shape.")


# ============================================================
# DEFAULT = AVENGERS
# ============================================================

current_pos = np.copy(
    pos_avengers
)

target_pos = np.copy(
    pos_avengers
)

current_color = np.copy(
    color_avengers
)

target_color = np.copy(
    color_avengers
)


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# ============================================================
# FINGER COUNT
#
# Giu logic cu:
# tip.y < pip.y
#
# 0 -> AVENGERS
# 1 -> IRON MAN
# 2 -> CAPTAIN
# 3 -> SPIDER-MAN
# 4 -> THOR
# 5 -> HULK
# ============================================================

finger_history = deque(
    maxlen=7
)


def get_palm_center(
    hand
):
    ids = [
        0,
        5,
        9,
        13,
        17
    ]

    x = sum(
        hand[i].x
        for i in ids
    ) / len(ids)

    y = sum(
        hand[i].y
        for i in ids
    ) / len(ids)

    return x, y


def point_distance_xy(
    x1,
    y1,
    x2,
    y2
):
    return math.sqrt(
        (x1 - x2) ** 2
        +
        (y1 - y2) ** 2
    )


# ============================================================
# 4 NGON CHINH
# index, middle, ring, pinky
# ============================================================

def get_four_finger_states(
    hand
):
    tips = [
        8,
        12,
        16,
        20
    ]

    pips = [
        6,
        10,
        14,
        18
    ]

    states = []

    for tip, pip in zip(
        tips,
        pips
    ):
        is_up = (
            hand[tip].y
            <
            hand[pip].y
        )

        states.append(
            is_up
        )

    return states


# ============================================================
# THUMB
#
# Chi kiem tra thumb khi 4 ngon kia deu mo.
# Nho vay nam dam se LUON la 0.
# ============================================================

def is_thumb_open_for_five(
    hand
):
    palm_x, palm_y = get_palm_center(
        hand
    )

    thumb_tip = hand[4]
    thumb_ip = hand[3]
    thumb_mcp = hand[2]

    tip_to_palm = point_distance_xy(
        thumb_tip.x,
        thumb_tip.y,
        palm_x,
        palm_y
    )

    ip_to_palm = point_distance_xy(
        thumb_ip.x,
        thumb_ip.y,
        palm_x,
        palm_y
    )

    tip_to_mcp = point_distance_xy(
        thumb_tip.x,
        thumb_tip.y,
        thumb_mcp.x,
        thumb_mcp.y
    )

    ip_to_mcp = point_distance_xy(
        thumb_ip.x,
        thumb_ip.y,
        thumb_mcp.x,
        thumb_mcp.y
    )

    return (
        tip_to_palm
        >
        ip_to_palm * 1.15

        and

        tip_to_mcp
        >
        ip_to_mcp * 1.30
    )


# ============================================================
# COUNT FINGERS
# ============================================================

def count_fingers_up(
    hand
):
    states = get_four_finger_states(
        hand
    )

    non_thumb_count = sum(
        states
    )

    # 4 ngon chinh deu gap -> LUON = 0
    if non_thumb_count == 0:
        return 0

    if non_thumb_count == 1:
        return 1

    if non_thumb_count == 2:
        return 2

    if non_thumb_count == 3:
        return 3

    # Chi phan biet 4 / 5 khi 4 ngon chinh deu mo
    if non_thumb_count == 4:
        if is_thumb_open_for_five(
            hand
        ):
            return 5

        return 4

    return 0


# ============================================================
# MODE
# ============================================================

def compute_gesture_mode(
    all_hands
):
    if not all_hands:
        finger_history.clear()
        return 1, 0

    hand = all_hands[0]

    raw_count = count_fingers_up(
        hand
    )

    finger_history.append(
        raw_count
    )

    # Median cua 7 frame de tranh nhay lung tung
    stable_count = int(
        np.median(
            list(
                finger_history
            )
        )
    )

    if stable_count == 0:
        return 1, 0

    elif stable_count == 1:
        return 2, 1

    elif stable_count == 2:
        return 3, 2

    elif stable_count == 3:
        return 4, 3

    elif stable_count == 4:
        return 5, 4

    else:
        return 6, 5


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_thread_func():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(
            "Khong mo duoc camera!"
        )

        shared_data[
            "running"
        ] = False

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    time.sleep(
        0.5
    )

    start_time = time.monotonic()
    last_timestamp = -1

    while shared_data[
        "running"
    ]:
        success, frame = cap.read()

        if not success:
            time.sleep(
                0.01
            )
            continue

        # Mirror
        frame = cv2.flip(
            frame,
            1
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(
            (
                time.monotonic()
                -
                start_time
            )
            *
            1000
        )

        if (
            timestamp_ms
            <=
            last_timestamp
        ):
            timestamp_ms = (
                last_timestamp
                +
                1
            )

        last_timestamp = timestamp_ms

        results = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Default = Avengers
        local_mode = 1
        local_finger_count = 0
        local_x = 0.0
        local_y = 0.0
        local_z = -12.0

        if results.hand_landmarks:
            all_hands = (
                results.hand_landmarks
            )

            (
                local_mode,
                local_finger_count
            ) = compute_gesture_mode(
                all_hands
            )

            hand = all_hands[0]

            wrist = hand[0]

            local_x = (
                wrist.x
                -
                0.5
            ) * 8.0

            local_y = -(
                wrist.y
                -
                0.5
            ) * 5.5

            pinky_mcp = hand[17]

            hand_size = math.sqrt(
                (
                    wrist.x
                    -
                    pinky_mcp.x
                ) ** 2
                +
                (
                    wrist.y
                    -
                    pinky_mcp.y
                ) ** 2
            )

            local_z = (
                -10.5
                -
                (
                    1.0
                    /
                    (
                        hand_size
                        +
                        0.01
                    )
                )
                *
                0.15
            )

            h, w, _ = frame.shape

            for start, end in (
                HAND_CONNECTIONS
            ):
                lm1 = hand[start]
                lm2 = hand[end]

                p1 = (
                    int(
                        lm1.x
                        *
                        w
                    ),
                    int(
                        lm1.y
                        *
                        h
                    )
                )

                p2 = (
                    int(
                        lm2.x
                        *
                        w
                    ),
                    int(
                        lm2.y
                        *
                        h
                    )
                )

                cv2.line(
                    frame,
                    p1,
                    p2,
                    (255, 0, 255),
                    2
                )

            for landmark in hand:
                cx = int(
                    landmark.x
                    *
                    w
                )

                cy = int(
                    landmark.y
                    *
                    h
                )

                cv2.circle(
                    frame,
                    (
                        cx,
                        cy
                    ),
                    4,
                    (0, 255, 0),
                    -1
                )

            # Highlight 5 fingertips
            for tip_id in [
                4,
                8,
                12,
                16,
                20
            ]:
                tip = hand[
                    tip_id
                ]

                tx = int(
                    tip.x
                    *
                    w
                )

                ty = int(
                    tip.y
                    *
                    h
                )

                cv2.circle(
                    frame,
                    (
                        tx,
                        ty
                    ),
                    7,
                    (0, 0, 255),
                    2
                )

        else:
            finger_history.clear()

        with lock:
            shared_data[
                "mode"
            ] = local_mode

            shared_data[
                "finger_count"
            ] = local_finger_count

            shared_data[
                "target_x"
            ] = local_x

            shared_data[
                "target_y"
            ] = local_y

            shared_data[
                "target_z"
            ] = local_z

            shared_data[
                "frame"
            ] = frame

    cap.release()


# ============================================================
# START CAMERA
# ============================================================

camera_thread = threading.Thread(
    target=camera_thread_func,
    daemon=True
)

camera_thread.start()

time.sleep(
    0.8
)


# ============================================================
# PYGAME
# ============================================================

pygame.init()

pygame.display.set_mode(
    (
        WIDTH,
        HEIGHT
    ),
    DOUBLEBUF
    |
    OPENGL
)

pygame.display.set_caption(
    "Marvel Gesture Particle Controller"
)


# ============================================================
# OPENGL
# ============================================================

glMatrixMode(
    GL_PROJECTION
)

glLoadIdentity()

gluPerspective(
    45,
    WIDTH / HEIGHT,
    0.1,
    50.0
)

glMatrixMode(
    GL_MODELVIEW
)

glEnable(
    GL_DEPTH_TEST
)

glEnable(
    GL_BLEND
)

clock = pygame.time.Clock()

hand_x = 0.0
hand_y = 0.0
hand_z = -12.0

rotation_angle = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

while shared_data[
    "running"
]:
    for event in pygame.event.get():
        if (
            event.type
            ==
            pygame.QUIT
        ):
            shared_data[
                "running"
            ] = False

        elif (
            event.type
            ==
            KEYDOWN
            and
            event.key
            ==
            K_ESCAPE
        ):
            shared_data[
                "running"
            ] = False

    with lock:
        current_mode = (
            shared_data[
                "mode"
            ]
        )

        finger_count = (
            shared_data[
                "finger_count"
            ]
        )

        target_hand_x = (
            shared_data[
                "target_x"
            ]
        )

        target_hand_y = (
            shared_data[
                "target_y"
            ]
        )

        target_hand_z = (
            shared_data[
                "target_z"
            ]
        )

        frame = (
            shared_data[
                "frame"
            ]
        )

    # ========================================================
    # CAMERA MONITOR
    # ========================================================

    if frame is not None:
        mode_labels = {
            1: "AVENGERS",
            2: "IRON MAN",
            3: "CAPTAIN AMERICA",
            4: "SPIDER-MAN",
            5: "THOR",
            6: "HULK"
        }

        cv2.putText(
            frame,
            f"FINGERS: {finger_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"MODE: {mode_labels.get(current_mode, 'AVENGERS')}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "0 = AVENGERS",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "1 = IRON MAN",
            (10, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "2 = CAPTAIN",
            (10, 141),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "3 = SPIDER-MAN",
            (10, 164),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "4 = THOR",
            (10, 187),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "5 = HULK",
            (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "Hand Sensor Monitor",
            frame
        )

        if (
            cv2.waitKey(1)
            &
            0xFF
        ) == ord(
            "q"
        ):
            shared_data[
                "running"
            ] = False

    # ========================================================
    # SMOOTH POSITION
    # ========================================================

    hand_x += (
        target_hand_x
        -
        hand_x
    ) * 0.20

    hand_y += (
        target_hand_y
        -
        hand_y
    ) * 0.20

    hand_z += (
        target_hand_z
        -
        hand_z
    ) * 0.20

    # ========================================================
    # SELECT SHAPE
    # ========================================================

    if current_mode == 1:
        target_pos = pos_avengers
        target_color = color_avengers

    elif current_mode == 2:
        target_pos = pos_ironman
        target_color = color_ironman

    elif current_mode == 3:
        target_pos = pos_captain
        target_color = color_captain

    elif current_mode == 4:
        target_pos = pos_spiderman
        target_color = color_spiderman

    elif current_mode == 5:
        target_pos = pos_thor
        target_color = color_thor

    elif current_mode == 6:
        target_pos = pos_hulk
        target_color = color_hulk

    else:
        target_pos = pos_avengers
        target_color = color_avengers

    # ========================================================
    # MORPH
    # ========================================================

    current_pos += (
        target_pos
        -
        current_pos
    ) * 0.12

    current_color += (
        target_color
        -
        current_color
    ) * 0.10

    # ========================================================
    # CLEAR
    # ========================================================

    glClearColor(
        0.0,
        0.0,
        0.0,
        1.0
    )

    glClear(
        GL_COLOR_BUFFER_BIT
        |
        GL_DEPTH_BUFFER_BIT
    )

    glLoadIdentity()

    # ========================================================
    # POSITION
    # ========================================================

    glTranslatef(
        hand_x,
        hand_y,
        hand_z
    )

    # ========================================================
    # ROTATION
    # ========================================================

    if current_mode == 1:
        rotation_angle += 0.25

    elif current_mode == 3:
        rotation_angle += 0.45

    elif current_mode == 5:
        rotation_angle += 0.30

    else:
        rotation_angle *= 0.82

    glRotatef(
        rotation_angle,
        0.0,
        1.0,
        0.0
    )

    # ========================================================
    # HULK PULSE
    # ========================================================

    if current_mode == 6:
        pulse = (
            1.0
            +
            0.025
            *
            math.sin(
                time.time()
                *
                4.0
            )
        )

        glScalef(
            pulse,
            pulse,
            pulse
        )

    # ========================================================
    # THOR PULSE
    # ========================================================

    if current_mode == 5:
        thor_pulse = (
            1.0
            +
            0.015
            *
            math.sin(
                time.time()
                *
                10.0
            )
        )

        glScalef(
            thor_pulse,
            thor_pulse,
            thor_pulse
        )

    # ========================================================
    # PARTICLE STYLE
    # ========================================================

    glBlendFunc(
        GL_SRC_ALPHA,
        GL_ONE
    )

    if current_mode == 1:
        glPointSize(5.3)

    elif current_mode == 3:
        glPointSize(5.2)

    elif current_mode == 5:
        glPointSize(5.0)

    else:
        glPointSize(4.8)

    # ========================================================
    # DRAW PARTICLES
    # ========================================================

    glBegin(
        GL_POINTS
    )

    for i in range(
        NUM_PARTICLES
    ):
        r, g, b = current_color[i]

        glColor4f(
            float(r),
            float(g),
            float(b),
            0.95
        )

        glVertex3f(
            float(
                current_pos[
                    i,
                    0
                ]
            ),
            float(
                current_pos[
                    i,
                    1
                ]
            ),
            float(
                current_pos[
                    i,
                    2
                ]
            )
        )

    glEnd()

    pygame.display.flip()

    clock.tick(
        60
    )


# ============================================================
# CLEAN UP
# ============================================================

shared_data[
    "running"
] = False

camera_thread.join(
    timeout=1.0
)

detector.close()

cv2.destroyAllWindows()

pygame.quit()