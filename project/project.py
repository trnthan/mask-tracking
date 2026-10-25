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

# =========================
# MEDIAPIPE TASKS API
# =========================
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# LOCAL HAND LANDMARKER MODEL
# =========================

MODEL_PATH = r"model\hand_landmarker.task"

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


# =========================
# HSV -> RGB
# =========================

def hsv_to_rgb_np(h, s, v):
    """Vectorized HSV -> RGB. h, s, v are numpy arrays (or floats) in [0,1]."""

    h = np.asarray(h, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    i = np.floor(h * 6.0)
    f = h * 6.0 - i

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i = (i.astype(int)) % 6

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])

    return r, g, b


# =========================
# SETTINGS
# =========================

WIDTH, HEIGHT = 1000, 700
NUM_PARTICLES = 1500


# =========================
# SHARED DATA
# =========================

lock = threading.Lock()

shared_data = {
    "mode": 1,
    "target_x": 0.0,
    "target_y": 0.0,
    "target_z": -12.0,
    "frame": None,
    "running": True
}


# =========================
# SPACE PARTICLES
# =========================

pos_space = np.random.uniform(
    -4.0,
    4.0,
    (NUM_PARTICLES, 3)
)


# =========================
# SATURN / PLANET
# =========================

pos_planet = np.zeros(
    (NUM_PARTICLES, 3)
)

NUM_SPHERE = 700


for i in range(NUM_SPHERE):

    phi = np.random.uniform(
        0,
        2 * np.pi
    )

    costheta = np.random.uniform(
        -1,
        1
    )

    theta = np.arccos(costheta)

    r = 1.3

    pos_planet[i, 0] = (
        r
        * np.sin(theta)
        * np.cos(phi)
    )

    pos_planet[i, 1] = (
        r
        * np.sin(theta)
        * np.sin(phi)
    )

    pos_planet[i, 2] = (
        r
        * np.cos(theta)
    )


# Saturn rings

for i in range(
    NUM_SPHERE,
    NUM_PARTICLES
):

    theta = np.random.uniform(
        0,
        2 * np.pi
    )

    r = np.random.uniform(
        1.8,
        3.8
    )

    pos_planet[i, 0] = (
        r * np.cos(theta)
    )

    pos_planet[i, 1] = (
        np.random.uniform(
            -0.05,
            0.05
        )
    )

    pos_planet[i, 2] = (
        r * np.sin(theta)
    )


# =========================
# I LOVE YOU TEXT
# =========================

text_img = np.zeros(
    (200, 800),
    dtype=np.uint8
)

cv2.putText(
    text_img,
    "I LOVE YOU",
    (20, 130),
    cv2.FONT_HERSHEY_SIMPLEX,
    3.5,
    255,
    12,
    cv2.LINE_AA
)


y_indices, x_indices = np.where(
    text_img > 0
)

x_text = (
    x_indices - 400
) / 70.0

y_text = -(
    y_indices - 100
) / 70.0

z_text = np.random.uniform(
    -0.1,
    0.1,
    len(x_text)
)


pos_text = np.zeros(
    (NUM_PARTICLES, 3)
)

text_points = np.stack(
    (
        x_text,
        y_text,
        z_text
    ),
    axis=-1
)


x_norm_full = (
    x_text - x_text.min()
) / (
    x_text.max()
    - x_text.min()
    + 1e-6
)


chosen_indices = np.random.choice(
    len(text_points),
    NUM_PARTICLES
)

pos_text = text_points[
    chosen_indices
]

text_hue_base = x_norm_full[
    chosen_indices
]


# =========================
# GAY TEXT
# =========================

text_img2 = np.zeros(
    (200, 800),
    dtype=np.uint8
)

cv2.putText(
    text_img2,
    "GAY",
    (150, 130),
    cv2.FONT_HERSHEY_SIMPLEX,
    3.5,
    255,
    12,
    cv2.LINE_AA
)


y_indices2, x_indices2 = np.where(
    text_img2 > 0
)

x_text2 = (
    x_indices2 - 400
) / 70.0

y_text2 = -(
    y_indices2 - 100
) / 70.0

z_text2 = np.random.uniform(
    -0.1,
    0.1,
    len(x_text2)
)

text_points2 = np.stack(
    (
        x_text2,
        y_text2,
        z_text2
    ),
    axis=-1
)

chosen_indices2 = np.random.choice(
    len(text_points2),
    NUM_PARTICLES
)

pos_text2 = text_points2[
    chosen_indices2
]


# =========================
# HEART
# =========================

pos_heart = np.zeros(
    (NUM_PARTICLES, 3)
)

color_heart = np.zeros(
    (NUM_PARTICLES, 3)
)

NUM_HEART_OUTLINE = int(
    NUM_PARTICLES * 0.4
)


DEEP_RED = np.array([
    0.85,
    0.03,
    0.18
])

HOT_PINK = np.array([
    1.0,
    0.35,
    0.55
])

GLOW_WHITE = np.array([
    1.0,
    0.8,
    0.88
])


# Heart outline

for i in range(
    NUM_HEART_OUTLINE
):

    t = np.random.uniform(
        -np.pi,
        np.pi
    )

    x = 16 * (
        np.sin(t) ** 3
    )

    y = (
        13 * np.cos(t)
        - 5 * np.cos(2 * t)
        - 2 * np.cos(3 * t)
        - np.cos(4 * t)
    )

    px = (
        (x / 16.0)
        * 2.05
        + np.random.uniform(
            -0.035,
            0.035
        )
    )

    py = (
        (y / 16.0)
        * 2.05
        + 0.45
        + np.random.uniform(
            -0.035,
            0.035
        )
    )

    pz = np.random.uniform(
        -0.12,
        0.12
    )

    pos_heart[i] = [
        px,
        py,
        pz
    ]

    color_heart[i] = (
        GLOW_WHITE
        if np.random.uniform(
            0.0,
            1.0
        ) > 0.85
        else HOT_PINK
    )


# Heart inside

for i in range(
    NUM_HEART_OUTLINE,
    NUM_PARTICLES
):

    t = np.random.uniform(
        -np.pi,
        np.pi
    )

    frac = (
        np.random.uniform(
            0.0,
            1.0
        ) ** 0.6
    )

    x = 16 * (
        np.sin(t) ** 3
    )

    y = (
        13 * np.cos(t)
        - 5 * np.cos(2 * t)
        - 2 * np.cos(3 * t)
        - np.cos(4 * t)
    )

    px = (
        (x / 16.0)
        * 2.0
        * frac
    )

    py = (
        (y / 16.0)
        * 2.0
        * frac
        + 0.45
    )

    pz = np.random.uniform(
        -0.08,
        0.08
    )

    pos_heart[i] = [
        px,
        py,
        pz
    ]

    color_heart[i] = (
        DEEP_RED * (1 - frac)
        + HOT_PINK * frac
    )


# =========================
# CURRENT PARTICLE POSITION
# =========================

current_pos = np.copy(
    pos_space
)

target_pos = np.copy(
    pos_space
)


# ============================================================
# GESTURE DETECTION
#
# QUAN TRỌNG:
# MediaPipe Tasks trả landmarks dạng:
#
# hand_landmarks[8].x
#
# KHÔNG PHẢI:
#
# hand_landmarks.landmark[8].x
#
# ============================================================

def compute_gesture_mode(hand_landmarks):

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


    finger_up = [
        hand_landmarks[t].y
        <
        hand_landmarks[p].y

        for t, p in zip(
            tips,
            pips
        )
    ]


    # Thumb tip
    thumb_tip = hand_landmarks[4]

    # Index tip
    index_tip = hand_landmarks[8]


    distance_thumb_index = math.sqrt(
        (
            thumb_tip.x
            - index_tip.x
        ) ** 2
        +
        (
            thumb_tip.y
            - index_tip.y
        ) ** 2
    )


    others_curled = (
        not finger_up[1]
        and not finger_up[2]
        and not finger_up[3]
    )


    # =========================
    # MODE 3
    # Finger Heart
    # =========================

    if (
        distance_thumb_index < 0.06
        and others_curled
    ):
        return 3


    # =========================
    # MODE 5
    # Rock sign
    # =========================

    if (
        finger_up[0]
        and finger_up[3]
        and not finger_up[1]
        and not finger_up[2]
    ):
        return 5


    # =========================
    # MODE 4
    # Fist
    # =========================

    if sum(finger_up) == 0:
        return 4


    # =========================
    # MODE 2
    # Index only
    # =========================

    if (
        finger_up[0]
        and not any(
            finger_up[1:]
        )
    ):
        return 2


    # =========================
    # MODE 1
    # Open / default
    # =========================

    return 1


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_thread_func():

    cap = cv2.VideoCapture(0)

    time.sleep(0.5)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        480
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        360
    )


    # Timestamp bắt đầu
    start_time = time.monotonic()

    last_timestamp = -1


    while shared_data["running"]:

        ret, frame = cap.read()


        if not ret:

            time.sleep(0.01)

            continue


        # Mirror camera

        frame = cv2.flip(
            frame,
            1
        )


        # =========================
        # BGR -> RGB
        # =========================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # =========================
        # OpenCV -> MediaPipe Image
        # =========================

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # =========================
        # TIMESTAMP
        #
        # detect_for_video yêu cầu
        # timestamp tăng liên tục
        # =========================

        timestamp_ms = int(
            (
                time.monotonic()
                - start_time
            )
            * 1000
        )


        if timestamp_ms <= last_timestamp:

            timestamp_ms = (
                last_timestamp + 1
            )


        last_timestamp = timestamp_ms


        # =========================
        # LOCAL MODEL DETECTION
        # =========================

        results = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )


        local_mode = 1

        local_x = 0.0
        local_y = 0.0
        local_z = -12.0


        # =========================
        # HAND FOUND
        # =========================

        if results.hand_landmarks:

            for hand_landmarks in (
                results.hand_landmarks
            ):


                # =========================
                # GESTURE
                # =========================

                local_mode = (
                    compute_gesture_mode(
                        hand_landmarks
                    )
                )


                # =========================
                # WRIST
                # landmark 0
                # =========================

                wrist = hand_landmarks[0]


                local_x = (
                    wrist.x - 0.5
                ) * 10.0


                local_y = -(
                    wrist.y - 0.5
                ) * 7.0


                # =========================
                # DEPTH ESTIMATION
                #
                # wrist -> pinky MCP
                # =========================

                pinky_mcp = (
                    hand_landmarks[17]
                )


                distance = math.sqrt(

                    (
                        wrist.x
                        - pinky_mcp.x
                    ) ** 2

                    +

                    (
                        wrist.y
                        - pinky_mcp.y
                    ) ** 2
                )


                local_z = (
                    -10.0
                    -
                    (
                        1.0
                        /
                        (
                            distance
                            + 0.01
                        )
                    )
                    * 0.2
                )


                # =========================
                # DRAW HAND LANDMARKS
                # Không dùng mp.solutions
                # =========================

                h, w, _ = frame.shape


                # Vẽ 21 landmark

                for landmark in hand_landmarks:

                    cx = int(
                        landmark.x * w
                    )

                    cy = int(
                        landmark.y * h
                    )


                    cv2.circle(
                        frame,
                        (cx, cy),
                        3,
                        (0, 255, 0),
                        -1
                    )


                # =========================
                # HAND CONNECTIONS
                # Tự định nghĩa để hoàn toàn
                # không dùng Solutions API
                # =========================

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


                for start, end in HAND_CONNECTIONS:

                    p1 = (
                        int(
                            hand_landmarks[
                                start
                            ].x * w
                        ),
                        int(
                            hand_landmarks[
                                start
                            ].y * h
                        )
                    )


                    p2 = (
                        int(
                            hand_landmarks[
                                end
                            ].x * w
                        ),
                        int(
                            hand_landmarks[
                                end
                            ].y * h
                        )
                    )


                    cv2.line(
                        frame,
                        p1,
                        p2,
                        (255, 0, 255),
                        1
                    )


        # =========================
        # UPDATE SHARED DATA
        # =========================

        with lock:

            shared_data[
                "mode"
            ] = local_mode

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


# =========================
# START CAMERA THREAD
# =========================

camera_thread = threading.Thread(
    target=camera_thread_func,
    daemon=True
)

camera_thread.start()


time.sleep(1.0)


# =========================
# PYGAME INIT
# =========================

pygame.init()

pygame.display.set_mode(
    (WIDTH, HEIGHT),
    DOUBLEBUF | OPENGL
)

pygame.display.set_caption(
    "Space Gesture Controller - Local MediaPipe Model"
)


# =========================
# OPENGL
# =========================

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


clock = pygame.time.Clock()

rotation_angle = 0.0

hand_x = 0.0
hand_y = 0.0
hand_z = -12.0


# ============================================================
# MAIN LOOP
# ============================================================

while shared_data["running"]:

    pygame.event.pump()


    for event in pygame.event.get():

        if (
            event.type == pygame.QUIT
            or
            (
                event.type == KEYDOWN
                and
                event.key == K_ESCAPE
            )
        ):

            shared_data[
                "running"
            ] = False


    # =========================
    # READ CAMERA DATA
    # =========================

    with lock:

        current_mode = (
            shared_data["mode"]
        )

        target_hand_x = (
            shared_data["target_x"]
        )

        target_hand_y = (
            shared_data["target_y"]
        )

        target_hand_z = (
            shared_data["target_z"]
        )

        frame = (
            shared_data["frame"]
        )


    # =========================
    # CAMERA WINDOW
    # =========================

    if frame is not None:

        mode_labels = {

            1:
                "COSMOS ROTATION (Open Hand)",

            2:
                "SATURN 3D (Index Finger)",

            3:
                "I LOVE YOU TEXT (Finger Heart)",

            4:
                "HEART / LOVE SHAPE (Fist)",

            5:
                "GAY TEXT (Rock 'n' Roll Sign)"
        }


        cv2.putText(
            frame,
            f"MODE: {mode_labels[current_mode]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


        cv2.imshow(
            "Hand Sensor Monitor",
            frame
        )


        if (
            cv2.waitKey(1)
            & 0xFF
            ==
            ord("q")
        ):

            shared_data[
                "running"
            ] = False


    # =========================
    # CLEAR OPENGL
    # =========================

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


    # =========================
    # SMOOTH HAND MOVEMENT
    # =========================

    hand_x += (
        target_hand_x
        - hand_x
    ) * 0.25


    hand_y += (
        target_hand_y
        - hand_y
    ) * 0.25


    hand_z += (
        target_hand_z
        - hand_z
    ) * 0.25


    # =========================
    # MODE
    # =========================

    if current_mode == 1:

        target_pos = pos_space

        rotation_angle += 0.5


    elif current_mode == 2:

        target_pos = pos_planet

        rotation_angle += 2.0


    elif current_mode == 3:

        target_pos = pos_text

        rotation_angle = 0.0


    elif current_mode == 4:

        target_pos = pos_heart

        rotation_angle += 1.5


    elif current_mode == 5:

        target_pos = pos_text2

        rotation_angle = 0.0


    # =========================
    # MORPH
    # =========================

    current_pos += (
        target_pos
        - current_pos
    ) * 0.15


    # =========================
    # TIME
    # =========================

    t_now = time.time()


    # Rainbow I LOVE YOU

    if current_mode == 3:

        hue = (
            text_hue_base
            +
            t_now * 0.15
        ) % 1.0


        text_r, text_g, text_b = (
            hsv_to_rgb_np(
                hue,
                0.85,
                1.0
            )
        )


    # Heart pulse

    elif current_mode == 4:

        heart_pulse = (
            0.85
            +
            0.15
            *
            math.sin(
                t_now * 3.0
            )
        )


    # =========================
    # MOVE OBJECT WITH HAND
    # =========================

    if current_mode in [
        2,
        3,
        4,
        5
    ]:

        glTranslatef(
            hand_x,
            hand_y,
            hand_z
        )


        if current_mode == 2:

            glRotatef(
                25,
                1.0,
                0.0,
                0.5
            )


    else:

        glTranslatef(
            0.0,
            0.0,
            -12.0
        )


    # =========================
    # ROTATION
    # =========================

    glRotatef(
        rotation_angle,
        0.0,
        1.0,
        0.0
    )


    # =========================
    # BLENDING
    # =========================

    glEnable(
        GL_BLEND
    )


    if current_mode in (
        3,
        4
    ):

        glBlendFunc(
            GL_SRC_ALPHA,
            GL_ONE
        )

        glPointSize(
            5.5
        )


    else:

        glBlendFunc(
            GL_SRC_ALPHA,
            GL_ONE_MINUS_SRC_ALPHA
        )

        glPointSize(
            4.5
        )


    # =========================
    # DRAW PARTICLES
    # =========================

    glBegin(
        GL_POINTS
    )


    for i in range(
        NUM_PARTICLES
    ):


        if current_mode == 3:

            glColor4f(
                text_r[i],
                text_g[i],
                text_b[i],
                0.9
            )


        elif current_mode == 4:

            cr, cg, cb = (
                color_heart[i]
            )


            glColor4f(

                min(
                    cr * heart_pulse,
                    1.0
                ),

                min(
                    cg * heart_pulse,
                    1.0
                ),

                min(
                    cb * heart_pulse,
                    1.0
                ),

                0.9
            )


        elif current_mode == 5:

            glColor4f(
                1.0,
                0.3,
                0.0,
                0.95
            )


        elif (
            current_mode == 2
            and
            i >= NUM_SPHERE
        ):

            glColor4f(
                1.0,
                0.7,
                0.3,
                0.6
            )


        elif (
            current_mode == 2
            and
            i < NUM_SPHERE
        ):

            glColor4f(
                1.0,
                0.5,
                0.0,
                0.85
            )


        else:

            glColor4f(
                0.1,
                0.5,
                1.0,
                0.8
            )


        glVertex3f(
            current_pos[i, 0],
            current_pos[i, 1],
            current_pos[i, 2]
        )


    glEnd()


    pygame.display.flip()

    clock.tick(
        60
    )


# ============================================================
# CLEAN UP
# ============================================================

shared_data["running"] = False

camera_thread.join(
    timeout=1.0
)

detector.close()

cv2.destroyAllWindows()

pygame.quit()