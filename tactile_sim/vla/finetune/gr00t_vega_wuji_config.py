"""GR00T N1.7 NEW_EMBODIMENT modality config for the Vega U + WUJI Hand 2 sim demos (tactile_sim.vla).

Matches meta/modality.json of a dataset from tactile_sim.vla.lerobot_export:
  state  (27): right_arm (7) right_hand (20)
  action (27): right_arm (7) right_hand (20)
  video      : ego_view (the Vega head camera)   language: annotation.human.task_description

The right arm is RELATIVE (joint deltas from the current state, the N1.7 convention, as in the user's G1_INSPIRE
config); the hand is ABSOLUTE MIT position targets (its gains and feedforward come from the dataset's
meta/vega_wuji.json table, not from the model). Standalone on purpose: GR00T imports it by path.

    python gr00t/data/stats.py --dataset-path <ds> --embodiment-tag NEW_EMBODIMENT --modality-config-path <this>
    python gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.7-3B --dataset-path <ds> \
        --embodiment-tag NEW_EMBODIMENT --modality-config-path <this> ...
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ACTION_HORIZON = 16  # 0.53 s at 30 Hz: one hammer swing


def _rel():
    return ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)


def _abs():
    return ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)


vega_wuji_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["right_arm", "right_hand"]),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=["right_arm", "right_hand"],
        action_configs=[_rel(), _abs()],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(vega_wuji_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
