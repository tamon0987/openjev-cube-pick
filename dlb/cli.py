"""``dlb`` command line.

dlb backends                          list backend profiles
dlb health  --backend openjev         probe a server
dlb smoke   --backend typesafe        one request of each question type
dlb gen     --out data/v1 --episodes 40
dlb offline --backend typesafe --modality text  --dataset data/v1 --out results/offline
dlb offline --backend djev     --modality image --dataset data/v1 --out results/offline
dlb online  --backend oracle   --modality text  --episodes 20 --out results/online
dlb twotier --backend openjev --policy command --planner openai --episodes 5
dlb report  --results results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", required=True, help="profile name in configs/backends or a YAML path")
    p.add_argument("--base-url", default=None, help="override base_url")
    p.add_argument("--model", default=None, help="override model field")


def _build(args: argparse.Namespace):
    from dlb.backends import build_backend

    overrides = {}
    if args.base_url:
        overrides["base_url"] = args.base_url
    if args.model:
        overrides["model"] = args.model
    return build_backend(args.backend, **overrides)


def cmd_backends(_: argparse.Namespace) -> int:
    from dlb.backends import list_profiles, load_profile

    for name in list_profiles():
        cfg = load_profile(name)
        print(
            f"{name:14s} kind={cfg.get('kind', 'jev_http'):8s} url={cfg.get('base_url', '-')} images={cfg.get('image_policy', '-')}"
        )
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    be = _build(args)
    print(json.dumps(be.health(), indent=2))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """One request per question type. Also probes whether the server accepts images."""
    import numpy as np

    from dlb.contract import Choice, DecisionRequest, Noul, Score, image_to_data_url

    be = _build(args)
    qs = {
        "route": Choice(
            instructions="Which team should handle this?",
            criteria={
                "billing": "Payments and refunds",
                "support": "How-to questions",
                "engineering": "Bugs and outages",
            },
        ),
        "urgent": Noul(instructions="Does this need attention within the hour?"),
        "severity": Score(
            instructions="How severe is the issue?", criteria=["cosmetic", "degraded", "outage"]
        ),
    }
    req = DecisionRequest(state={"ticket": "Checkout is down for all customers since 10:02."}, questions=qs)
    resp = be.decide(req)
    print(
        f"[{be.info.name}] text request ok: {resp.latency_s * 1000:.0f} ms, model={resp.model}, usage={resp.usage}"
    )
    for k, a in resp.answers.items():
        print(f"  {k:9s} -> {a.label:12s} conf={a.conf:.2f}  {a.to_wire()}")
    if be.wants_images():
        img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype("uint8")
        img[16:48, 16:48] = [220, 30, 30]
        req2 = DecisionRequest(
            state={"note": "One photo attached."},
            questions={"red": Noul(instructions="Does the image contain a red square?")},
            images=[image_to_data_url(img)],
        )
        try:
            r2 = be.decide(req2)
            a = r2.answers["red"]
            print(
                f"[{be.info.name}] image request ok: {r2.latency_s * 1000:.0f} ms, p(red square)={a.noul:.2f}, usage={r2.usage}"
            )
        except Exception as e:
            print(f"[{be.info.name}] image request FAILED: {e}")
            print(
                "  -> check the server's image field name (configs/backends/*.yaml: image_field) or set image_policy: drop"
            )
            return 1
    return 0


def cmd_gen(args: argparse.Namespace) -> int:
    from dlb.eval.dataset import generate

    out = generate(
        args.out,
        n_episodes=args.episodes,
        epsilon=args.epsilon,
        seed=args.seed,
        image_size=args.image_size,
        cameras=tuple(args.cameras.split(",")),
    )
    print(json.dumps(json.load(open(out / "summary.json")), indent=2))
    return 0


def cmd_offline(args: argparse.Namespace) -> int:
    from dlb.eval.dataset import load
    from dlb.eval.offline import run_offline

    be = _build(args)
    samples = load(args.dataset, limit=args.limit)
    summary = run_offline(
        be,
        samples,
        modality=args.modality,
        question_set=args.question_set,
        out_dir=args.out,
        image_cameras=tuple(args.cameras.split(",")) if args.cameras else None,
        run_name=args.run_name,
    )
    q = summary["questions"]
    print(
        f"[{summary['run']}] n={summary['answered']} err={summary['errors']} "
        f"p50={summary['latency_s']['p50'] * 1000:.0f}ms "
        + " ".join(f"{k}={v['accuracy']:.3f}" for k, v in q.items())
    )
    return 0


def cmd_online(args: argparse.Namespace) -> int:
    from dlb.harness import EpisodeRunner, GatingConfig, summarize
    from dlb.sim.env import PickPlaceEnv

    be = _build(args)
    env = PickPlaceEnv(render=args.modality != "text", image_size=args.image_size, max_steps=args.max_steps)
    runner = EpisodeRunner(
        env,
        be,
        modality=args.modality,
        gating=GatingConfig(threshold=args.gate, on_low_confidence=args.on_low),
        log_dir=Path(args.out) / "logs" / (args.run_name or be.info.name),
        save_images=args.save_images,
    )
    results = []
    for i in range(args.episodes):
        r = runner.run(i, seed=args.seed + i)
        results.append(r)
        print(
            f"ep {i:3d} seed={r.seed} success={r.success} steps={r.steps} acc={r.action_accuracy:.2f} esc={r.escalations} {r.actions}"
        )
    s = summarize(results)
    s["run"] = args.run_name or f"{be.info.name}__{args.modality}"
    s["gate"] = {"threshold": args.gate, "on_low_confidence": args.on_low}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / f"{s['run']}.summary.json", "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    print(json.dumps(s, indent=2))
    return 0


def cmd_twotier(args: argparse.Namespace) -> int:
    from dlb.harness import twotier
    from dlb.harness.twotier import (
        JevBisectServoPolicy,
        JevCommandPolicy,
        JevPerceptionPolicy,
        JevServoPolicy,
        OpenAIPlanner,
        OraclePolicy,
        ScriptedPlanner,
        SequencePlanner,
        TwoTierRunner,
        summarize_twotier,
    )

    if args.object_names:
        cube, bin_ = (x.strip() for x in args.object_names.split(","))
        twotier.set_object_names(cube, bin_)
    from dlb.sim.env import PickPlaceEnv

    cams = tuple(args.cameras.split(","))
    if args.policy == "oracle":
        policy = OraclePolicy()
    else:
        kw = dict(
            cameras=cams,
            cross=not args.no_cross,
            black=args.black,
            show_hint=not args.no_hint,
            zoom=args.zoom,
        )
        if args.policy == "command":
            policy = JevCommandPolicy(_build(args), **kw)
        elif args.policy == "servo":
            policy = JevServoPolicy(_build(args), **kw)
        elif args.policy == "bisect":
            policy = JevBisectServoPolicy(_build(args), **kw)
        else:
            policy = JevPerceptionPolicy(_build(args), rotate_v=not args.no_rotate, **kw)
    planner = (
        OpenAIPlanner(model=args.planner_model, reasoning_effort=args.planner_effort or None)
        if args.planner == "openai"
        else SequencePlanner(
            step_cm=args.step_cm,
            fine_cm=args.fine_cm,
            search_cube=args.search_cube,
            search_bin=args.search_bin,
            overhead=args.overhead_guide,
        )
        if args.planner == "sequence"
        else ScriptedPlanner(
            step_cm=args.step_cm,
            fine_cm=args.fine_cm,
            hints=args.planner_hints,
            corrections=args.planner_corrections,
        )
    )
    run = args.run_name or f"{planner.name}__{policy.name}"
    if args.robot == "sim":
        env = PickPlaceEnv(cameras=("front", "wrist"), image_size=args.image_size)
    else:
        from dlb.real.omx import RealOMX

        # mock / real: motion goes through ros2_control; "mock" renders the twin's virtual scene
        env = RealOMX(
            args.robot_config,
            camera_source="twin" if args.robot == "mock" else "usb",
            cameras=("front", "wrist"),
            image_size=args.image_size,
        )
    guide = None
    if args.overhead_guide:
        from dlb.harness.marking import OverheadGuide, OverheadMarker

        names = (
            tuple(x.strip() for x in args.object_names.split(","))
            if args.object_names
            else ("orange cube", "black bin")
        )
        guide = OverheadGuide(
            OverheadMarker(model=args.mark_model, n=args.mark_n, object_names=names),
            log_dir=Path(args.out) / "logs" / run / "marks",
        )
    runner = TwoTierRunner(
        env,
        planner,
        policy,
        max_decisions=args.max_decisions,
        conf_threshold=args.gate,
        log_dir=Path(args.out) / "logs" / run,
        save_images=not args.no_save_images,
        guide=guide,
    )
    results = []
    try:
        for i in range(args.episodes):
            r = runner.run(i, seed=args.seed + i)
            results.append(r)
            print(
                f"ep {i:3d} seed={r.seed} success={r.success} decisions={r.decisions} planner={r.planner_calls} "
                f"esc={r.escalations} miss={r.missed_grasps} pre={r.pre_moves} acc={r.command_accuracy:.2f} "
                f"{r.stop_reason} {r.subtasks}",
                flush=True,
            )
    finally:
        if hasattr(env, "close"):  # the real robot's ROS and camera threads (the simulator has none)
            env.close()
    s = summarize_twotier(results)
    s["run"] = run
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / f"{run}.summary.json", "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    print(json.dumps(s, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from dlb.eval.report import build_report

    md = build_report(args.results, out_file=args.out)
    print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="dlb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("backends").set_defaults(fn=cmd_backends)

    p = sub.add_parser("health")
    _add_backend_args(p)
    p.set_defaults(fn=cmd_health)

    p = sub.add_parser("smoke")
    _add_backend_args(p)
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("gen")
    p.add_argument("--out", default="data/v1")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--epsilon", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--cameras", default="front,wrist")
    p.set_defaults(fn=cmd_gen)

    p = sub.add_parser("offline")
    _add_backend_args(p)
    p.add_argument("--dataset", default="data/v1")
    p.add_argument("--modality", choices=["text", "image", "image+text"], default="text")
    p.add_argument("--question-set", choices=["action_only", "full"], default="full")
    p.add_argument("--cameras", default=None, help="comma list subset of dataset cameras")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="results/offline")
    p.add_argument("--run-name", default=None)
    p.set_defaults(fn=cmd_offline)

    p = sub.add_parser("online")
    _add_backend_args(p)
    p.add_argument("--modality", choices=["text", "image", "image+text"], default="text")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--gate", type=float, default=0.0, help="confidence threshold; 0 disables gating")
    p.add_argument("--on-low", choices=["act", "oracle", "home", "stop"], default="act")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--out", default="results/online")
    p.add_argument("--run-name", default=None)
    p.set_defaults(fn=cmd_online)

    p = sub.add_parser("twotier", help="planner (slow) + decision layer picking relative moves (fast)")
    _add_backend_args(p)
    p.add_argument(
        "--policy", choices=["command", "perception", "servo", "bisect", "oracle"], default="command"
    )
    p.add_argument(
        "--planner",
        choices=["scripted", "openai", "sequence"],
        default="scripted",
        help="scripted: simulator truth; sequence: fixed order without truth (real robot); openai: VLM",
    )
    p.add_argument("--object-names", default=None, help='e.g. "orange cube,black bin" (prompts say red/blue)')
    p.add_argument(
        "--search-cube", default="", help="sequence planner: wrist-image move while the cube is out of view"
    )
    p.add_argument(
        "--search-bin", default="", help="sequence planner: wrist-image move while the bin is out of view"
    )
    p.add_argument(
        "--overhead-guide",
        action="store_true",
        help="sequence planner: coarse moves to the cube and the bin by VLM marks in the overhead image",
    )
    p.add_argument("--mark-model", default="gpt-5.5", help="vision model that marks the overhead image")
    p.add_argument("--mark-n", type=int, default=5, help="parallel marking queries (median)")
    p.add_argument("--planner-model", default="gpt-5.5")
    p.add_argument("--planner-effort", default="low", help="reasoning effort; empty string to omit")
    p.add_argument("--cameras", default="wrist", help="cameras the decision layer sees")
    p.add_argument("--no-cross", action="store_true", help="do not draw the below-gripper cross")
    p.add_argument("--black", action="store_true", help="ablation: send black images")
    p.add_argument("--no-hint", action="store_true", help="do not pass the planner's hint")
    p.add_argument(
        "--zoom", action="store_true", help="also send a zoomed crop around the below-gripper cross"
    )
    p.add_argument(
        "--no-rotate", action="store_true", help="perception: do not re-ask top-bottom on a rotated view"
    )
    p.add_argument("--step-cm", type=float, default=2.0, help="scripted planner coarse step")
    p.add_argument("--fine-cm", type=float, default=1.0, help="scripted planner fine step near the cube")
    p.add_argument("--planner-hints", action="store_true", help="scripted planner adds corrective hints")
    p.add_argument(
        "--planner-corrections",
        action="store_true",
        help="scripted planner sends truth-based corrective moves after failures",
    )
    p.add_argument("--gate", type=float, default=0.0, help="escalate below this confidence; 0 disables")
    p.add_argument(
        "--robot",
        choices=["sim", "mock", "real"],
        default="sim",
        help="sim: MuJoCo only; mock/real: motion through ros2_control (mock renders the twin)",
    )
    p.add_argument("--robot-config", default="configs/robot/omx_f.yaml")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--max-decisions", type=int, default=80)
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--no-save-images", action="store_true")
    p.add_argument("--out", default="results/twotier")
    p.add_argument("--run-name", default=None)
    p.set_defaults(fn=cmd_twotier)

    p = sub.add_parser("report")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    if args.cmd in ("gen", "online") and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
