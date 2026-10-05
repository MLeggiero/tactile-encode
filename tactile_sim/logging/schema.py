"""Episode file layout (HDF5; npz fallback with the same keys flattened by '/').

/                  attrs: schema_version, config_json, mujoco_version, git_sha, seed, dt_phys, arm_source, created
/physics/          ground truth at the physics rate / logging.truth_decim:
                   t, face_pos[N,3], face_vel[N,3], nail_depth, contact_force, pad_forces[N,2],
                   hammer_in_hand_pos[N,3], hammer_in_hand_rotvec[N,3], qfrc_constraint_arm[N,7]
/sensors/<name>/   t_sample, t_avail, value[M,dim], seq            (native sensor rate, after the sensor model)
/control/          L1 at 1 kHz: t, tau, q, qd, x[7], x_eq[7], xd_eq[6], K[6], F_ff[6], F_grip, mode, impact_flag,
                   gated, tau_ext[7], f_ext[6]
/grip/             grip loop at 500 Hz: t, setpoint, measured, command
/events            compound (name S32, t f8)
/strikes           compound STRIKE_DTYPE, one row per strike
/summary           attrs: episode summary
"""

SCHEMA_VERSION = "1.0"
GROUPS = ("physics", "sensors", "control", "grip")
