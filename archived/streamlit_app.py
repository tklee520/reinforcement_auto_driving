#!/usr/bin/env python3
"""Streamlit dashboard for the V3 safety highway PPO script.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DEFAULT_BACKEND = APP_DIR / "train_highway_longtail_ppo_realistic_v3_safety_web.py"
DEFAULT_MODEL_DIR = APP_DIR / "models"
DEFAULT_LOG_DIR = APP_DIR / "runs" / "ppo_highway_longtail_realistic_v3_safety"
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
    st.markdown("**将要执行的命令：**")
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
        status_box.info(f"任务运行中：{elapsed:.1f} 秒，最后输出 {len(lines)} 行")
        output_box.code("\n".join(lines[-240:]) or "等待输出...", language="text")

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
        status_box.success(f"完成：{run_name}，耗时 {elapsed:.1f} 秒")
    else:
        status_box.error(f"失败：return code = {return_code}。请看上面的日志。")
    return return_code


def common_runtime_controls(prefix: str) -> dict:
    st.subheader("环境与安全参数")
    col1, col2, col3 = st.columns(3)
    with col1:
        traffic_mode = st.selectbox(
            "Traffic mode",
            ["simple", "realistic", "dense"],
            index=1,
            key=f"{prefix}_traffic_mode",
            help="simple 更容易；dense 更拥挤、更难。",
        )
        seed = st.number_input("Seed", min_value=0, max_value=999_999, value=42, step=1, key=f"{prefix}_seed")
        duration = st.number_input(
            "视频/episode 长度 duration",
            min_value=10,
            max_value=300,
            value=80,
            step=5,
            key=f"{prefix}_duration",
            help="单位是仿真秒；RecordVideo 会按每个 evaluation episode 录制一个 mp4。",
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
            max_value=2.4,
            value=1.55,
            step=0.05,
            key=f"{prefix}_fast_lane_safety_multiplier",
            help="越大越保守，快车道需要更长跟车距离。",
        )
        fast_lane_min_ttc = st.slider(
            "Fast-lane minimum TTC",
            min_value=1.0,
            max_value=7.0,
            value=3.2,
            step=0.1,
            key=f"{prefix}_fast_lane_min_ttc",
            help="越大越早触发减速/阻止加速。",
        )
    with col3:
        fast_lane_accel_gap_multiplier = st.slider(
            "Fast-lane accel gap multiplier",
            min_value=1.0,
            max_value=2.2,
            value=1.35,
            step=0.05,
            key=f"{prefix}_fast_lane_accel_gap_multiplier",
            help="快车道前方间距不足时阻止 FASTER。",
        )
        continuous_action = st.checkbox("Continuous action", value=False, key=f"{prefix}_continuous_action")
        debug_eval = st.checkbox("Debug evaluation 输出动作统计", value=True, key=f"{prefix}_debug_eval")

    col4, col5, col6 = st.columns(3)
    with col4:
        enable_overtake_assist = st.checkbox("启用 overtake assist", value=True, key=f"{prefix}_enable_overtake_assist")
    with col5:
        enable_train_noise = st.checkbox("训练时启用 action noise", value=True, key=f"{prefix}_enable_train_noise")
    with col6:
        eval_action_noise = st.checkbox("Eval 时启用 action noise", value=False, key=f"{prefix}_eval_action_noise")

    return {
        "traffic_mode": traffic_mode,
        "seed": int(seed),
        "duration": int(duration),
        "eval_weather": float(eval_weather),
        "fast_lane_safety_multiplier": float(fast_lane_safety_multiplier),
        "fast_lane_min_ttc": float(fast_lane_min_ttc),
        "fast_lane_accel_gap_multiplier": float(fast_lane_accel_gap_multiplier),
        "continuous_action": bool(continuous_action),
        "debug_eval": bool(debug_eval),
        "enable_overtake_assist": bool(enable_overtake_assist),
        "enable_train_noise": bool(enable_train_noise),
        "eval_action_noise": bool(eval_action_noise),
    }


def ppo_controls(prefix: str) -> dict:
    st.subheader("训练参数")
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
    st.subheader("Evaluation / 视频参数")
    col1, col2, col3 = st.columns(3)
    with col1:
        eval_episodes = st.number_input(
            "录制几个 evaluation 视频",
            min_value=1,
            max_value=30,
            value=3,
            step=1,
            key=f"{prefix}_eval_episodes",
            help="每个 episode 会生成一个 mp4，所以这里就是视频数量。",
        )
    with col2:
        record_video = st.checkbox("录制并保存视频", value=True, key=f"{prefix}_record_video")
    with col3:
        render_mode = st.selectbox("Render mode", ["rgb_array", "human"], index=0, key=f"{prefix}_render_mode", help="网页录制建议用 rgb_array。")
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
    ]
    if params["continuous_action"]:
        cmd.append("--continuous-action")
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


def show_videos(paths: list[Path], title: str = "视频") -> None:
    st.subheader(title)
    if not paths:
        st.info("还没有找到 mp4 视频。先运行一次录制 evaluation。")
        return
    for video in paths:
        stat = video.stat()
        rel = video.relative_to(APP_DIR) if video.is_relative_to(APP_DIR) else video
        with st.expander(f"{rel} · {human_size(stat.st_size)} · {dt.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}", expanded=False):
            st.video(str(video))
            with video.open("rb") as f:
                st.download_button("下载这个视频", data=f, file_name=video.name, mime="video/mp4", key=f"download_{video}")


def main() -> None:
    st.set_page_config(page_title="Highway PPO V3 Safety Dashboard", layout="wide")
    st.title("Highway PPO V3 Safety 参数控制台")
    st.caption("用 Streamlit 调参、训练、evaluation 录制，以及集中查看生成的视频。")

    with st.sidebar:
        st.header("路径设置")
        backend_path = to_abs_path(st.text_input("V3 backend script", value=str(DEFAULT_BACKEND)))
        model_dir = to_abs_path(st.text_input("Model dir", value=str(DEFAULT_MODEL_DIR)))
        log_dir = to_abs_path(st.text_input("Log dir", value=str(DEFAULT_LOG_DIR)))
        video_root = to_abs_path(st.text_input("Video library dir", value=str(DEFAULT_VIDEO_DIR)))
        st.divider()
        st.write(f"Backend exists: {'✅' if backend_path.exists() else '❌'}")
        st.write(f"Models found: {len(list_models(model_dir))}")
        st.write(f"Videos found: {len(list_videos(video_root))}")
        st.caption("路径可以是绝对路径，也可以是相对这个 app 文件夹的相对路径。")

    paths = {"backend_path": backend_path, "model_dir": model_dir, "log_dir": log_dir, "video_root": video_root}
    if not backend_path.exists():
        st.error(f"找不到 backend script：{backend_path}")
        st.stop()

    tab_train, tab_eval, tab_gallery, tab_notes = st.tabs(["训练 + 录视频", "只做 Evaluation", "视频库", "参数建议"])

    with tab_train:
        st.markdown("这里会启动一次训练；训练完成后，如果没有勾选 `skip eval`，会自动按 eval episodes 录制视频。")
        common = common_runtime_controls("train")
        train_params = ppo_controls("train")
        eval_params = eval_controls("train")
        skip_eval = st.checkbox("只训练，不跑 evaluation", value=False, key="train_skip_eval")

        if st.button("开始训练 / 训练后录视频", type="primary", key="start_train"):
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
                show_videos(list_videos(run_video_dir), title="本次生成的视频")

    with tab_eval:
        st.markdown("选择一个已有 `.zip` 模型，直接跑 evaluation 并录制视频。")
        models = list_models(model_dir)
        if not models:
            st.warning("当前 model dir 里没有找到 .zip 模型。可以先在第一个 tab 训练，或者把 model dir 指到已有模型文件夹。")
        model_options = [str(p) for p in models]
        selected_model = st.selectbox("选择模型", model_options, index=0 if model_options else None, key="eval_selected_model") if model_options else ""

        common_eval = common_runtime_controls("eval")
        eval_only_params = eval_controls("eval")
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0, key="eval_device")
        common_eval["device"] = device

        if st.button("开始 evaluation 并录视频", type="primary", key="start_eval", disabled=not bool(selected_model)):
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
                show_videos(list_videos(run_video_dir), title="本次生成的视频")

    with tab_gallery:
        videos = list_videos(video_root)
        st.markdown(f"当前视频库：`{video_root}`")
        col1, col2 = st.columns([1, 3])
        with col1:
            max_show = st.number_input("最多显示几个", min_value=1, max_value=100, value=20, step=1, key="gallery_max_show")
        show_videos(videos[: int(max_show)], title="最近生成的视频")

    with tab_notes:
        st.subheader("推荐调参范围")
        st.markdown(
            """
- `eval episodes`：录几个 evaluation 视频就填几。比如 5 会生成 5 个 mp4。
- `duration`：控制每个 episode 的仿真时长，也就是视频大概长度。建议先用 40–80 快速看效果，再用 120+ 做最终展示。
- `fast-lane safety multiplier`：默认 1.55。追尾多就调高到 1.7；太保守就降到 1.3–1.45。
- `fast-lane minimum TTC`：默认 3.2。越大越早刹车，越安全但可能更怂。
- `fast-lane accel gap multiplier`：默认 1.35。越大越不容易在快车道继续加速。
- `timesteps`：快速试验 50k–100k；正式训练 300k+；dense traffic 可以 400k+。
- `PPO n-steps`：默认 2048 比较稳。显存/内存有限可以降到 1024。
- `traffic mode`：先用 realistic；效果稳定后再用 dense 做压力测试。
            """
        )
        st.info("训练和 evaluation 是同步执行的；运行时不要重复点击按钮。长训练建议开一个专用终端运行 Streamlit。")


if __name__ == "__main__":
    main()
