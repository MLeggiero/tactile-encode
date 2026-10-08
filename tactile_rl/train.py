"""Train on a tactile task with mjlab's RSL-RL runner (its CLI, with our tasks registered):

    python -m tactile_rl.train Tactile-Hold-VegaWuji-1000Hz --env.scene.num-envs 4096 --agent.max-iterations 100
"""

from __future__ import annotations


def main() -> None:
    from mjlab.scripts.train import main as mjlab_train

    import tactile_rl.tasks  # noqa: F401 - registers the tasks

    mjlab_train()


if __name__ == "__main__":
    main()
