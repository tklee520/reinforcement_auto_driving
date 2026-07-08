# Highway PPO V3 Safety Streamlit Dashboard

这个文件夹给 `V3 safety` 版本加了一个 Streamlit 网页控制台，方便你：

1. 在网页上挑 PPO 训练参数：`timesteps`、`n-steps`、`batch-size`、`learning-rate`、`n-envs` 等。
2. 在网页上挑 V3 safety 参数：`fast-lane safety multiplier`、`fast-lane minimum TTC`、`fast-lane accel gap multiplier`。
3. 设置 evaluation 视频数量：`eval episodes` 填几，就会录几个 mp4。
4. 设置视频/episode 长度：`duration`，单位是仿真秒。
5. 训练后自动 evaluation 并录视频。
6. 只选择已有模型做 evaluation 并录视频。
7. 在网页的“视频库”里直接查看和下载生成的视频。

## 文件说明

- `streamlit_app.py`：网页控制台。
- `train_highway_longtail_ppo_realistic_v3_safety_web.py`：基于你的 V3 safety 版本做的小改版。
  - 新增了 `--duration`，用于控制每个 episode / 视频长度。
  - 新增了 `--model-path`，方便网页端直接选择某个 `.zip` 模型 evaluation。
- `requirements.txt`：建议安装的依赖。
- `run_streamlit.sh` / `run_streamlit.bat`：启动脚本。

## 安装依赖

建议在你的项目环境或虚拟环境里运行：

```bash
cd highway_v3_streamlit
pip install -r requirements.txt
```

如果你已经装过 `stable-baselines3`、`highway-env`、`torch`，也可以只补装：

```bash
pip install streamlit moviepy imageio-ffmpeg
```

## 启动网页

macOS / Linux：

```bash
cd highway_v3_streamlit
./run_streamlit.sh
```

Windows：双击 `run_streamlit.bat`，或者运行：

```bat
cd highway_v3_streamlit
streamlit run streamlit_app.py
```

## 使用建议

### 快速试验

- `timesteps`: 50000 到 100000
- `duration`: 40 到 80
- `eval episodes`: 2 到 3
- `traffic mode`: realistic

### 正式训练

- `timesteps`: 300000+
- `duration`: 80 到 120
- `eval episodes`: 5+
- `traffic mode`: realistic 或 dense

### 安全参数怎么调

- 如果视频里还是有追尾风险：
  - 把 `fast-lane safety multiplier` 从 1.55 调到 1.7。
  - 或把 `fast-lane minimum TTC` 从 3.2 调到 3.8。
- 如果车太保守、不敢加速或不敢超车：
  - 把 `fast-lane safety multiplier` 降到 1.3 到 1.45。
  - 或把 `fast-lane accel gap multiplier` 从 1.35 降到 1.2。

## 输出位置

默认会在这个文件夹下生成：

```text
models/   # 模型 .zip
runs/     # 日志、eval logs、monitor logs
videos/   # 每次录制的视频，按 run timestamp 分文件夹
```

网页里的“视频库”会递归扫描 `videos/` 里面所有 `.mp4`，可以直接播放和下载。

## 注意

训练和 evaluation 会在 Streamlit 当前进程里同步运行。训练较久时，不要重复点击按钮。长训练建议开一个专用终端运行网页。
