"""Real robot: run only reach -> align -> descend_grasp with the Jev servo, then stop holding the cube.

uv run python scripts/real_grasp_only.py --object-names "orange cube,black bin"
"""

from __future__ import annotations

import argparse

from dlb.backends import build_backend
from dlb.harness import twotier
from dlb.harness.twotier import TASK_ORDER, JevServoPolicy, Plan, SequencePlanner, TwoTierRunner


class GraspOnly(SequencePlanner):
    name = "grasp_only"

    def plan(self, env, images, events):
        last = events[-1] if events else None
        if last is not None and last["outcome"] == "queue_finished":
            return Plan(task_complete=True, subtasks=[], scene="grasp done")
        return Plan(task_complete=False, subtasks=[self._sub(s) for s in TASK_ORDER[:3]], scene="grasp only")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-names", default="orange cube,black bin")
    ap.add_argument("--max-decisions", type=int, default=40)
    ap.add_argument("--run-name", default="real_grasp")
    a = ap.parse_args()
    twotier.set_object_names(*(x.strip() for x in a.object_names.split(",")))
    from dlb.real.omx import RealOMX

    env = RealOMX("configs/robot/omx_f.yaml", camera_source="usb", cameras=("front", "wrist"), image_size=320)
    policy = JevServoPolicy(build_backend("openjev"), cameras=("wrist",))
    runner = TwoTierRunner(
        env,
        GraspOnly(),
        policy,
        max_decisions=a.max_decisions,
        max_planner_calls=4,
        log_dir=f"results/twotier/logs/{a.run_name}",
    )
    r = runner.run(0, seed=0)
    print(
        "decisions",
        r.decisions,
        "subtasks",
        r.subtasks,
        "missed",
        r.missed_grasps,
        "stop",
        r.stop_reason,
        "held",
        env.held,
        "gripper",
        round(env.measured()[1], 3),
    )
    env.close()


if __name__ == "__main__":
    main()
