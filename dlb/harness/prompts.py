"""Question builders: turn an Observation into a DecisionRequest.

Question keys here must match ``PickPlaceEnv.oracle_labels`` keys so that every
answer can be scored against ground truth.

Two "question sets" are provided:

* ``action_only`` – one ``choice`` question (next primitive). Cheapest; what the
  online control loop needs.
* ``full``        – next action + ``holding``/``aligned``/``task_complete`` nouls
  + a ``progress`` score. Used for the offline dataset so we can measure each
  sub-skill (grasp detection, alignment, phase estimation) separately.
"""

from __future__ import annotations

import json
from typing import Any

from dlb.contract import Choice, DecisionRequest, Noul, Question, Score, image_to_data_url
from dlb.sim.env import PRIMITIVES, TASK_TEXT, Observation

SYSTEM_PREAMBLE = (
    "You are the decision layer of a robot arm controller. Each cycle you see the "
    "current scene and must pick exactly one motion primitive. Primitives are executed "
    "by a deterministic controller; you never output joint angles. "
    "Typical sequence: hover_object -> descend -> grasp -> lift -> hover_bin -> release -> done. "
    "If the gripper is closed but not holding the cube, choose release. "
    "If a step failed, repeat the appropriate primitive rather than skipping ahead."
)

PROGRESS_LEVELS = [
    "Gripper is not above the cube and nothing is held.",
    "Gripper is aligned directly above (or at) the cube, nothing is held yet.",
    "Cube is held in the gripper, gripper is not yet above the bin.",
    "Cube is held and the gripper is above the bin, ready to release.",
    "Cube rests inside the bin and is no longer held; the task is complete.",
]


def next_action_question() -> Choice:
    return Choice(
        instructions="Which primitive should the controller execute next?", criteria=dict(PRIMITIVES)
    )


def questions(question_set: str = "action_only") -> dict[str, Question]:
    qs: dict[str, Question] = {"next_action": next_action_question()}
    if question_set == "full":
        qs["holding"] = Noul(
            instructions="Is the red cube currently held between the gripper fingers?",
            criteria={
                "true": "The jaw is closed around the cube and the cube moves with the gripper.",
                "false": "The cube is on the table, in the bin, or the jaw is open/closed on nothing.",
            },
        )
        qs["aligned"] = Noul(
            instructions="Is the gripper positioned directly above the cube (xy offset under 1.2 cm)?",
        )
        qs["task_complete"] = Noul(
            instructions="Is the task complete: cube inside the bin and gripper open?",
        )
        qs["progress"] = Score(
            instructions="Which stage best describes the current scene?", criteria=PROGRESS_LEVELS
        )
    elif question_set != "action_only":
        raise ValueError(f"unknown question set {question_set!r}")
    return qs


def build_request(
    obs: Observation,
    question_set: str = "action_only",
    modality: str = "text",
    image_cameras: tuple[str, ...] | None = None,
    include_state_with_images: bool = False,
    oracle: dict[str, Any] | None = None,
) -> DecisionRequest:
    """
    modality:
      ``text``        – state JSON only (TypeSafe Jev)
      ``image``       – images + minimal text (task, primitive semantics, history), no geometry
      ``image+text``  – images + full state JSON (upper bound for image backends)
    """
    qs = questions(question_set)
    history = obs.state_json.get("history", {})
    images: list[str] = []
    if modality in ("image", "image+text"):
        cams = image_cameras or tuple(obs.images.keys())
        for cam in cams:
            if cam not in obs.images:
                raise KeyError(f"camera {cam!r} not rendered; available: {list(obs.images)}")
            images.append(image_to_data_url(obs.images[cam]))
    if modality == "text" or modality == "image+text" or include_state_with_images:
        state: str | dict[str, Any] = {"instructions": SYSTEM_PREAMBLE, **obs.state_json}
    elif modality == "image":
        cams = image_cameras or tuple(obs.images.keys())
        state = {
            "instructions": SYSTEM_PREAMBLE,
            "task": TASK_TEXT,
            "images": [f"image {i + 1}: camera '{c}'" for i, c in enumerate(cams)]
            + ["The wrist camera looks down between the two dark gripper fingers."],
            "history": history,
        }
    else:
        raise ValueError(f"unknown modality {modality!r}")
    meta: dict[str, Any] = {"modality": modality, "question_set": question_set, "step": obs.step}
    if oracle is not None:
        meta["oracle"] = oracle
    return DecisionRequest(state=state, questions=qs, images=images, meta=meta)


def state_to_text(state: dict[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))
