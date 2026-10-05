"""MJCF composition. Everything here produces XML; nothing owns simulation state."""

from tactile_sim.model.builder import SceneSpec, build_scene, strike_axis, tcp_rotation

__all__ = ["SceneSpec", "build_scene", "strike_axis", "tcp_rotation"]
