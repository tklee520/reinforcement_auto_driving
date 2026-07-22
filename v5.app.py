#!/usr/bin/env python3
"""Streamlit dashboard for the V5 overtaking-logic highway PPO script.

Run:
    streamlit run v5.app.py
"""
from __future__ import annotations

import datetime as dt
import json
import hashlib
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import streamlit as st

APP_DIR = Path(__file__).resolve().parent


def find_default_backend() -> Path:
    """Find the V5 backend even when the downloaded file has a suffix such as ``(5)``."""
    canonical = APP_DIR / "train_highway_longtail_ppo_realistic_v5_overtake_logic_web.py"
    if canonical.exists():
        return canonical

    candidates = sorted(
        APP_DIR.glob("train_highway_longtail_ppo_realistic_v5_overtake_logic_web*.py"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else canonical


DEFAULT_BACKEND = find_default_backend()
DEFAULT_MODEL_DIR = APP_DIR / "models"
DEFAULT_LOG_DIR = APP_DIR / "runs" / "ppo_highway_longtail_realistic_v5_overtake_logic"
DEFAULT_VIDEO_DIR = APP_DIR / "videos"
RUN_HISTORY_FILE = APP_DIR / "streamlit_run_history.jsonl"


def quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def timestamp_run_name(prefix: str) -> str:
    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}"


def to_abs_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (APP_DIR / path).resolve()
    return path


def list_models(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []
    return sorted(model_dir.rglob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)


def list_videos(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        return []
    return sorted(video_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def append_history(record: dict) -> None:
    try:
        with RUN_HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def run_subprocess(cmd: list[str], cwd: Path, run_name: str, expected_video_dir: Path | None = None) -> int:
    st.markdown("**Command to be executed:**")
    st.code(quote_cmd(cmd), language="bash")

    output_box = st.empty()
    status_box = st.empty()
    start_time = time.time()

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    lines: list[str] = []
    assert process.stdout is not None
    for line in iter(process.stdout.readline, ""):
        if not line:
            break
        lines.append(line.rstrip())
        elapsed = time.time() - start_time
        status_box.info(f"Task running: {elapsed:.1f} seconds elapsed, {len(lines)} output lines received")
        output_box.code("\n".join(lines[-240:]) or "Waiting for output...", language="text")

    return_code = process.wait()
    elapsed = time.time() - start_time

    record = {
        "run_name": run_name,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "return_code": return_code,
        "elapsed_seconds": round(elapsed, 2),
        "command": cmd,
        "video_dir": str(expected_video_dir) if expected_video_dir else None,
    }
    append_history(record)

    if return_code == 0:
        status_box.success(f"Completed: {run_name} in {elapsed:.1f} seconds")
    else:
        status_box.error(f"Failed: return code = {return_code}. Check the log above.")
    return return_code


def common_runtime_controls(prefix: str) -> dict:
    st.subheader("Environment and Safety Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        traffic_mode = st.selectbox(
            "Traffic mode",
            ["simple", "realistic", "dense"],
            index=1,
            key=f"{prefix}_traffic_mode",
            help="Simple is easier; dense has heavier traffic and is more difficult. Use realistic first, then try dense after the system is stable.",
        )
        seed = st.number_input("Seed", min_value=0, max_value=999_999, value=42, step=1, key=f"{prefix}_seed")
        duration = st.number_input(
            "Episode Duration (simulation seconds)",
            min_value=10,
            max_value=1000,
            value=80,
            step=5,
            key=f"{prefix}_duration",
            help="Measured in simulation seconds. RecordVideo creates one MP4 for each evaluation episode.",
        )
    with col2:
        eval_weather = st.slider(
            "Eval weather / rain intensity",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
            key=f"{prefix}_eval_weather",
        )
        fast_lane_safety_multiplier = st.slider(
            "Fast-lane safety multiplier",
            min_value=1.0,
            max_value=2.6,
            value=1.75,
            step=0.05,
            key=f"{prefix}_fast_lane_safety_multiplier",
            help="Higher values are more conservative and require a longer following distance in the fast lane.",
        )
        fast_lane_min_ttc = st.slider(
            "Fast-lane minimum TTC",
            min_value=1.0,
            max_value=8.0,
            value=4.0,
            step=0.1,
            key=f"{prefix}_fast_lane_min_ttc",
            help="Higher values trigger braking or acceleration blocking earlier. Increase this if rear-end risks remain.",
        )
    with col3:
        fast_lane_accel_gap_multiplier = st.slider(
            "Fast-lane accel gap multiplier",
            min_value=1.0,
            max_value=2.5,
            value=1.60,
            step=0.05,
            key=f"{prefix}_fast_lane_accel_gap_multiplier",
            help="Blocks the FASTER action when the front gap in the fast lane is insufficient.",
        )
        continuous_action = st.checkbox("Continuous action", value=False, key=f"{prefix}_continuous_action")
        debug_eval = st.checkbox("Debug evaluation and action statistics", value=True, key=f"{prefix}_debug_eval")

    st.subheader("Rear-End Collision Prevention")
    col4, col5, col6 = st.columns(3)
    with col4:
        enable_speed_governor = st.checkbox(
            "Enable speed governor",
            value=True,
            key=f"{prefix}_enable_speed_governor",
            help="Recommended. It directly reduces ego.target_speed when the front gap is insufficient, preventing continued closing even when the action is IDLE.",
        )
        enable_target_lane_safety_check = st.checkbox(
            "Check front vehicle in target lane",
            value=True,
            key=f"{prefix}_target_lane_safety_check",
            help="Recommended. The system checks vehicles ahead in both the current lane and the target lane during a lane change.",
        )
    with col5:
        speed_governor_gap_multiplier = st.slider(
            "Speed-governor gap multiplier",
            min_value=1.0,
            max_value=2.8,
            value=1.90,
            step=0.05,
            key=f"{prefix}_speed_governor_gap_multiplier",
            help="The speed governor activates when the front gap falls below this multiplier times the safe distance. Increase it for earlier intervention.",
        )
        fast_lane_speed_margin = st.slider(
            "Fast-lane speed margin",
            min_value=-1.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            key=f"{prefix}_fast_lane_speed_margin",
            help="Maximum amount by which the target speed may exceed the front vehicle speed in an unsafe fast-lane situation. Zero means no positive speed margin.",
        )
    with col6:
        normal_lane_speed_margin = st.slider(
            "Normal-lane speed margin",
            min_value=0.0,
            max_value=3.0,
            value=1.2,
            step=0.1,
            key=f"{prefix}_normal_lane_speed_margin",
            help="Maximum amount by which the target speed may exceed the front vehicle speed in an unsafe normal-lane situation.",
        )

    st.subheader("Purposeful Lane-Change Filter")
    col_lc1, col_lc2, col_lc3 = st.columns(3)
    with col_lc1:
        useful_lane_change_only = st.checkbox(
            "Allow only purposeful lane changes",
            value=True,
            key=f"{prefix}_useful_lane_change_only",
            help="Recommended. The vehicle moves left only when the current lane is blocked and the left lane offers a clear benefit. It returns right only when the right lane will remain usable.",
        )
    with col_lc2:
        lane_change_min_benefit = st.slider(
            "Lane-change min benefit (m)",
            min_value=5.0,
            max_value=60.0,
            value=22.0,
            step=1.0,
            key=f"{prefix}_lane_change_min_benefit",
            help="Minimum additional clear distance required in the target lane before a left lane change is considered worthwhile. Higher values reduce unnecessary lane changes.",
        )
    with col_lc3:
        return_right_clear_time = st.slider(
            "Return-right clear time (s)",
            min_value=3.0,
            max_value=20.0,
            value=10.0,
            step=0.5,
            key=f"{prefix}_return_right_clear_time",
            help="If the vehicle is expected to encounter another slow vehicle soon after returning right, it delays the return to avoid repeated weaving.",
        )

    col7, col8, col9 = st.columns(3)
    with col7:
        enable_overtake_assist = st.checkbox("Enable overtake assist", value=True, key=f"{prefix}_enable_overtake_assist")
    with col8:
        enable_train_noise = st.checkbox("Enable action noise during training", value=True, key=f"{prefix}_enable_train_noise")
    with col9:
        eval_action_noise = st.checkbox("Enable action noise during evaluation", value=False, key=f"{prefix}_eval_action_noise")

    return {
        "traffic_mode": traffic_mode,
        "seed": int(seed),
        "duration": int(duration),
        "eval_weather": float(eval_weather),
        "fast_lane_safety_multiplier": float(fast_lane_safety_multiplier),
        "fast_lane_min_ttc": float(fast_lane_min_ttc),
        "fast_lane_accel_gap_multiplier": float(fast_lane_accel_gap_multiplier),
        "enable_speed_governor": bool(enable_speed_governor),
        "enable_target_lane_safety_check": bool(enable_target_lane_safety_check),
        "speed_governor_gap_multiplier": float(speed_governor_gap_multiplier),
        "fast_lane_speed_margin": float(fast_lane_speed_margin),
        "normal_lane_speed_margin": float(normal_lane_speed_margin),
        "useful_lane_change_only": bool(useful_lane_change_only),
        "lane_change_min_benefit": float(lane_change_min_benefit),
        "return_right_clear_time": float(return_right_clear_time),
        "continuous_action": bool(continuous_action),
        "debug_eval": bool(debug_eval),
        "enable_overtake_assist": bool(enable_overtake_assist),
        "enable_train_noise": bool(enable_train_noise),
        "eval_action_noise": bool(eval_action_noise),
    }


def ppo_controls(prefix: str) -> dict:
    st.subheader("Training Parameters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        timesteps = st.number_input("Total timesteps", min_value=1_000, max_value=5_000_000, value=300_000, step=10_000, key=f"{prefix}_timesteps")
        n_envs = st.number_input("n-envs", min_value=1, max_value=16, value=1, step=1, key=f"{prefix}_n_envs")
    with col2:
        learning_rate = st.number_input("Learning rate", min_value=1e-6, max_value=1e-2, value=3e-4, step=1e-5, format="%.6f", key=f"{prefix}_learning_rate")
        n_steps = st.number_input("PPO n-steps", min_value=64, max_value=16384, value=2048, step=64, key=f"{prefix}_n_steps")
    with col3:
        batch_size = st.number_input("Batch size", min_value=16, max_value=2048, value=64, step=16, key=f"{prefix}_batch_size")
        n_epochs = st.number_input("n-epochs", min_value=1, max_value=40, value=10, step=1, key=f"{prefix}_n_epochs")
    with col4:
        ent_coef = st.number_input("Entropy coef", min_value=0.0, max_value=0.1, value=0.0, step=0.001, format="%.4f", key=f"{prefix}_ent_coef")
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0, key=f"{prefix}_device")

    col5, col6, col7 = st.columns(3)
    with col5:
        eval_freq = st.number_input("Eval freq", min_value=1_000, max_value=500_000, value=10_000, step=1_000, key=f"{prefix}_eval_freq")
    with col6:
        n_eval_episodes = st.number_input("EvalCallback episodes", min_value=1, max_value=20, value=5, step=1, key=f"{prefix}_n_eval_episodes")
    with col7:
        checkpoint_freq = st.number_input("Checkpoint freq", min_value=1_000, max_value=1_000_000, value=25_000, step=1_000, key=f"{prefix}_checkpoint_freq")

    return {
        "timesteps": int(timesteps),
        "n_envs": int(n_envs),
        "learning_rate": float(learning_rate),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "ent_coef": float(ent_coef),
        "device": str(device),
        "eval_freq": int(eval_freq),
        "n_eval_episodes": int(n_eval_episodes),
        "checkpoint_freq": int(checkpoint_freq),
    }


def eval_controls(prefix: str) -> dict:
    st.subheader("Evaluation and Video Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        eval_episodes = st.number_input(
            "Number of evaluation videos",
            min_value=1,
            max_value=30,
            value=3,
            step=1,
            key=f"{prefix}_eval_episodes",
            help="Each evaluation episode generates one MP4, so this is also the number of videos.",
        )
    with col2:
        record_video = st.checkbox("Record and save videos", value=True, key=f"{prefix}_record_video")
    with col3:
        render_mode = st.selectbox("Render mode", ["rgb_array", "human"], index=0, key=f"{prefix}_render_mode", help="Use rgb_array for recording videos in the Streamlit app.")
    return {"eval_episodes": int(eval_episodes), "record_video": bool(record_video), "render_mode": str(render_mode)}


def add_common_cli_args(cmd: list[str], params: dict, paths: dict) -> None:
    cmd += [
        "--seed", str(params["seed"]),
        "--device", str(params.get("device", "auto")),
        "--log-dir", str(paths["log_dir"]),
        "--model-dir", str(paths["model_dir"]),
        "--traffic-mode", params["traffic_mode"],
        "--duration", str(params["duration"]),
        "--eval-weather", str(params["eval_weather"]),
        "--fast-lane-safety-multiplier", str(params["fast_lane_safety_multiplier"]),
        "--fast-lane-min-ttc", str(params["fast_lane_min_ttc"]),
        "--fast-lane-accel-gap-multiplier", str(params["fast_lane_accel_gap_multiplier"]),
        "--speed-governor-gap-multiplier", str(params["speed_governor_gap_multiplier"]),
        "--fast-lane-speed-margin", str(params["fast_lane_speed_margin"]),
        "--normal-lane-speed-margin", str(params["normal_lane_speed_margin"]),
        "--lane-change-min-benefit", str(params["lane_change_min_benefit"]),
        "--return-right-clear-time", str(params["return_right_clear_time"]),
    ]
    if params["continuous_action"]:
        cmd.append("--continuous-action")
    if not params["enable_speed_governor"]:
        cmd.append("--no-speed-governor")
    if not params["enable_target_lane_safety_check"]:
        cmd.append("--no-target-lane-safety-check")
    if not params["useful_lane_change_only"]:
        cmd.append("--allow-nonpurpose-lane-changes")
    if not params["enable_overtake_assist"]:
        cmd.append("--no-overtake-assist")
    if not params["enable_train_noise"]:
        cmd.append("--no-action-noise")
    if params["eval_action_noise"]:
        cmd.append("--eval-action-noise")
    if params["debug_eval"]:
        cmd.append("--debug-eval")


def add_ppo_cli_args(cmd: list[str], train_params: dict) -> None:
    cmd += [
        "--timesteps", str(train_params["timesteps"]),
        "--n-envs", str(train_params["n_envs"]),
        "--learning-rate", str(train_params["learning_rate"]),
        "--n-steps", str(train_params["n_steps"]),
        "--batch-size", str(train_params["batch_size"]),
        "--n-epochs", str(train_params["n_epochs"]),
        "--ent-coef", str(train_params["ent_coef"]),
        "--eval-freq", str(train_params["eval_freq"]),
        "--n-eval-episodes", str(train_params["n_eval_episodes"]),
        "--checkpoint-freq", str(train_params["checkpoint_freq"]),
    ]


def show_videos(paths: list[Path], title: str = "Videos") -> None:
    st.subheader(title)
    if not paths:
        st.info("No MP4 videos were found. Run an evaluation with video recording first.")
        return

    unique_paths: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        video = Path(raw_path)
        try:
            resolved = str(video.resolve())
        except OSError:
            resolved = str(video)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(video)

    for i, video in enumerate(unique_paths):
        try:
            stat = video.stat()
        except OSError:
            continue
        rel = video.relative_to(APP_DIR) if video.is_relative_to(APP_DIR) else video
        unique_key_seed = f"{title}|{i}|{video.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
        unique_key = hashlib.md5(unique_key_seed.encode("utf-8")).hexdigest()
        with st.expander(
            f"{rel} · {human_size(stat.st_size)} · {dt.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}",
            expanded=False,
        ):
            st.video(str(video))
            with video.open("rb") as f:
                st.download_button(
                    "Download this video",
                    data=f,
                    file_name=video.name,
                    mime="video/mp4",
                    key=f"download_{unique_key}",
                )


def main() -> None:
    st.set_page_config(page_title="Highway PPO V5 Dashboard", layout="wide")
    st.title("Highway PPO V5 Overtaking Logic Dashboard")
    st.caption("Configure parameters, train the PPO agent, run evaluation recordings, and review generated videos. V5 combines a speed governor with purposeful lane-change filtering to reduce unnecessary weaving.")

    with st.sidebar:
        st.header("Path Settings")
        backend_path = to_abs_path(st.text_input("V5 backend script", value=str(DEFAULT_BACKEND)))
        model_dir = to_abs_path(st.text_input("Model dir", value=str(DEFAULT_MODEL_DIR)))
        log_dir = to_abs_path(st.text_input("Log dir", value=str(DEFAULT_LOG_DIR)))
        video_root = to_abs_path(st.text_input("Video library dir", value=str(DEFAULT_VIDEO_DIR)))
        st.divider()
        st.write(f"Backend exists: {'✅' if backend_path.exists() else '❌'}")
        st.write(f"Models found: {len(list_models(model_dir))}")
        st.write(f"Videos found: {len(list_videos(video_root))}")
        st.caption("Paths may be absolute or relative to the folder containing this app.")

    paths = {"backend_path": backend_path, "model_dir": model_dir, "log_dir": log_dir, "video_root": video_root}
    if not backend_path.exists():
        st.error(f"Backend script not found: {backend_path}")
        st.stop()

    tab_train, tab_eval, tab_gallery, tab_notes = st.tabs(["Train and Record", "Evaluation Only", "Video Library", "Parameter Guide"])

    with tab_train:
        st.markdown("This tab starts a new training run. After training, videos are recorded automatically unless `Skip final evaluation` is selected. The V5 speed governor and purposeful lane-change filter are enabled by default.")
        common = common_runtime_controls("train")
        train_params = ppo_controls("train")
        eval_params = eval_controls("train")
        skip_eval = st.checkbox("Train only; skip final evaluation", value=False, key="train_skip_eval")

        if st.button("Start Training and Record Videos", type="primary", key="start_train"):
            run_name = timestamp_run_name("train_eval")
            run_video_dir = video_root / run_name if eval_params["record_video"] and not skip_eval else None
            if run_video_dir:
                run_video_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            cmd = [sys.executable, str(backend_path)]
            merged = {**common, **train_params}
            add_common_cli_args(cmd, merged, paths)
            add_ppo_cli_args(cmd, train_params)
            if skip_eval:
                cmd.append("--skip-eval")
            else:
                cmd += ["--eval-episodes", str(eval_params["eval_episodes"]), "--render-mode", eval_params["render_mode"]]
                if run_video_dir:
                    cmd += ["--video-dir", str(run_video_dir)]

            rc = run_subprocess(cmd, cwd=APP_DIR, run_name=run_name, expected_video_dir=run_video_dir)
            if rc == 0 and run_video_dir:
                show_videos(list_videos(run_video_dir), title="Videos Generated in This Run")

    with tab_eval:
        st.markdown("Select an existing `.zip` model to run evaluation and record videos without retraining.")
        models = list_models(model_dir)
        if not models:
            st.warning("No `.zip` model was found in the current model directory. Train a model in the first tab or point the model directory to an existing model folder.")
        model_options = [str(p) for p in models]
        selected_model = st.selectbox("Select Model", model_options, index=0 if model_options else None, key="eval_selected_model") if model_options else ""

        common_eval = common_runtime_controls("eval")
        eval_only_params = eval_controls("eval")
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0, key="eval_device")
        common_eval["device"] = device

        if st.button("Start Evaluation and Record Videos", type="primary", key="start_eval", disabled=not bool(selected_model)):
            run_name = timestamp_run_name("eval_only")
            run_video_dir = video_root / run_name if eval_only_params["record_video"] else None
            if run_video_dir:
                run_video_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(backend_path), "--skip-train", "--model-path", str(selected_model)]
            add_common_cli_args(cmd, common_eval, paths)
            cmd += ["--eval-episodes", str(eval_only_params["eval_episodes"]), "--render-mode", eval_only_params["render_mode"]]
            if run_video_dir:
                cmd += ["--video-dir", str(run_video_dir)]

            rc = run_subprocess(cmd, cwd=APP_DIR, run_name=run_name, expected_video_dir=run_video_dir)
            if rc == 0 and run_video_dir:
                show_videos(list_videos(run_video_dir), title="Videos Generated in This Run")

    with tab_gallery:
        videos = list_videos(video_root)
        st.markdown(f"Current video library: `{video_root}`")
        col1, col2 = st.columns([1, 3])
        with col1:
            max_show = st.number_input("Maximum videos to display", min_value=1, max_value=100, value=20, step=1, key="gallery_max_show")
        show_videos(videos[: int(max_show)], title="Recently Generated Videos")

    with tab_notes:
        st.subheader("Recommended Parameter Ranges")
        st.markdown(
            """
- `Evaluation episodes`: Enter the number of evaluation videos required. For example, 5 creates 5 MP4 files.
- `Duration`: Controls the simulation time of each episode and therefore the approximate video length. Use 40–80 for quick tests and 120 or more for a longer final demonstration.
- `Fast-lane safety multiplier`: Default 1.75. Increase to 1.9–2.1 for more conservative following; reduce to 1.55–1.65 if the vehicle is too cautious.
- `Fast-lane minimum TTC`: Default 4.0 seconds. Higher values trigger earlier braking but may reduce speed.
- `Fast-lane accel gap multiplier`: Default 1.60. Higher values make acceleration in the fast lane more restrictive.
- `Speed-governor gap multiplier`: Default 1.90. It directly limits `ego.target_speed`, addressing the case where an IDLE action still leaves a high target speed.
- `Fast-lane speed margin`: Default 0.0 m/s. In an unsafe fast-lane situation, zero prevents the target speed from exceeding the front vehicle speed. Use 0.3–0.8 for a more aggressive setting.
- `Check front vehicle in target lane`: Keep this enabled so that both the current and target lanes are checked during a lane change.
- If rear-end risks remain, increase the speed-governor gap multiplier, fast-lane safety multiplier, and fast-lane minimum TTC before switching to dense traffic.
- If the vehicle is too conservative, reduce the fast-lane safety multiplier to 1.55–1.65 or increase the fast-lane speed margin to about 0.3 m/s.
            """
        )



if __name__ == "__main__":
    main()
