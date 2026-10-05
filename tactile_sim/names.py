"""Canonical MJCF names shared by the model builder, sensors, controllers and tests."""

ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
ARM_BODIES = [f"fr3_link{i}" for i in range(0, 8)]
ARM_BASE_BODY = "fr3_link0"
ARM_MOTORS = [f"fr3_motor{i}" for i in range(1, 8)]
ATTACHMENT_SITE = "attachment_site"

# Wrist, hand and pads
FT_BODY = "ft_sensor_body"
FT_SITE = "ft_site"
WRIST_FLEX_BODY = "wrist_flex"
WRIST_FLEX_JOINTS = ["wrist_flex_x", "wrist_flex_y", "wrist_flex_z"]
HAND_BODY = "hand"
TCP_SITE = "tcp_site"
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
FINGER_BODIES = ["left_finger", "right_finger"]
GRIP_TENDON = "grip_split"
GRIP_MOTOR = "grip_motor"
PAD_BODIES = ["pad_L", "pad_R"]
PAD_GEOMS = ["pad_L_geom", "pad_R_geom"]
PAD_IMU_SITES = ["pad_imu_L", "pad_imu_R"]
PAD_JOINTS = {
    "pad_L": ["pad_L_n", "pad_L_t1", "pad_L_t2"],
    "pad_R": ["pad_R_n", "pad_R_t1", "pad_R_t2"],
}

# Hammer
HAMMER_BODY = "hammer"
HAMMER_FACE_GEOM = "hammer_face"
HAMMER_HEAD_GEOM = "hammer_head"
HAMMER_HANDLE_GEOM = "hammer_handle"
HAMMER_FACE_SITE = "hammer_face_site"
HAMMER_REF_SITE = "hammer_ref"
HAMMER_IMU_SITE = "hammer_imu"
GRASP_WELD = "grasp_weld"

# Nail plant
BOARD_BODY = "board"
BOARD_GEOM = "board_geom"
NAIL_BODY = "nail"
NAIL_JOINT = "nail_slide"
NAIL_HEAD_GEOM = "nail_head"
NAIL_HEAD_SITE = "nail_head_top"
