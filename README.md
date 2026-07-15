# Highway PPO V5：安全超车与长尾交通场景

> 基于 `highway-env` 与 Stable-Baselines3 PPO 的高速公路驾驶项目。最终版本采用 **PPO 策略 + 安全约束层 + 有目的换道逻辑**，重点解决追尾、无意义换道、超车道长期占用和复杂交通下的稳定性问题。

## English summary

A highway-driving project built with `highway-env` and Stable-Baselines3 PPO. The final V5 combines a learned PPO policy with a speed governor, a safety shield, weather-aware following distance, and purposeful overtaking logic.

## 项目特点

- PPO 输出离散驾驶动作：`LANE_LEFT`、`IDLE`、`LANE_RIGHT`、`FASTER`、`SLOWER`
- 3 秒换道冷却，防止连续左右摇摆
- 根据天气强度动态调整安全距离和 TTC 阈值
- 对当前车道和目标车道进行前车风险检查
- Speed governor 直接限制 `ego.target_speed`，避免 `IDLE` 状态下仍高速追尾
- 仅在存在明确收益时允许主动向左超车
- 超车结束后，在右侧车道可用时返回右侧
- 支持 `simple`、`realistic`、`dense` 三种交通场景
- 支持训练、仅 Evaluation、视频录制、模型选择和 Streamlit 参数控制

## 项目定位

本项目不是“完全无规则的纯 PPO”。最终系统结构为：

```text
PPO policy
    ↓
Overtake / lane-purpose filtering
    ↓
Safety shield + speed governor
    ↓
Highway environment
```

PPO 负责输出驾驶动作；规则层负责过滤明显危险或缺乏目的的动作，并在追尾风险较高时限制目标速度。答辩或报告中建议表述为：

> 本项目采用 PPO 学习驾驶策略，并结合基于安全距离、TTC、目标车道状态和超车收益的安全辅助层，提高长尾交通场景中的安全性与稳定性。

## 目录结构

```text
.
├── train_highway_ppo_v5.py       # PPO 训练、Evaluation 和视频录制后端
├── streamlit_app.py              # Streamlit 控制台
├── requirements.txt              # Python 依赖
├── MODEL_CARD.md                  # 模型说明与实验记录模板
├── CHANGELOG.md                   # 版本演进说明
├── docs/
│   ├── ARCHITECTURE.md            # 系统架构
│   └── EVALUATION.md              # 公平对比与视频录制指南
├── scripts/
│   ├── run_dashboard.sh           # 启动 Streamlit
│   └── evaluate_example.sh        # Evaluation 示例
├── models/                        # 训练模型，默认不提交 Git
├── videos/                        # Streamlit 生成的视频，默认不提交 Git
├── comparison_videos/             # 版本对比视频，默认不提交 Git
└── runs/                          # TensorBoard、评估和监控日志
```

## 环境要求

建议：

- Python 3.10 或 3.11
- macOS、Linux 或 Windows
- CPU 可以运行；训练速度较慢时可使用 CUDA GPU

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 最快使用方式：Streamlit

```bash
streamlit run streamlit_app.py
```

Streamlit 提供四个页面：

1. 训练并录制视频
2. 加载已有 `.zip` 模型，只做 Evaluation
3. 查看视频库
4. 参数建议

## 从头训练

```bash
python train_highway_ppo_v5.py \
  --timesteps 300000 \
  --traffic-mode realistic \
  --duration 200 \
  --progress-bar
```

训练结果默认保存在：

```text
models/
├── best/best_model.zip
├── checkpoints/
└── ppo_highway_longtail_realistic_v5_overtake_logic_discrete_realistic.zip
```

建议优先评估 `models/best/best_model.zip`，因为训练结束时保存的 final model 不一定是评估表现最好的 checkpoint。

## 加载已有模型，只做 Evaluation

```bash
python train_highway_ppo_v5.py \
  --skip-train \
  --model-path "models/best/best_model.zip" \
  --seed 42 \
  --traffic-mode realistic \
  --duration 720 \
  --eval-weather 0.15 \
  --eval-episodes 3 \
  --render-mode rgb_array \
  --video-fps 30 \
  --video-dir "comparison_videos/v5_seed42" \
  --debug-eval
```

生成的视频位于：

```text
comparison_videos/v5_seed42/
```

## 视频长度说明

`duration` 是 episode 的最长仿真时间，不是 MP4 的实际播放秒数；碰撞或驶出道路会提前终止。

本项目的 `policy_frequency=5`，RecordVideo 大致每个环境 step 记录一帧，因此完整 episode 的近似视频长度为：

```text
视频秒数 ≈ duration × 5 ÷ video_fps
```

示例：

| duration | video_fps | 理论视频长度 |
|---:|---:|---:|
| 720 | 30 | 约 120 秒 |
| 600 | 10 | 约 300 秒 |
| 300 | 5 | 约 300 秒 |

`video_fps` 只改变 MP4 播放速度，不改变 PPO 决策频率或驾驶行为。

## 推荐的最终演示参数

下面是一套偏安全、仍能出现超车动作的参考配置：

```text
Traffic mode                       realistic
Seed                               42
Duration                           720
Eval weather                       0.15
Fast-lane safety multiplier        1.90
Fast-lane minimum TTC              4.50
Fast-lane accel gap multiplier     1.75
Speed governor                     ON
Target-lane safety check           ON
Speed-governor gap multiplier      2.05
Fast-lane speed margin             0.00
Normal-lane speed margin           0.20
Purposeful lane changes only       ON
Lane-change min benefit            26 m
Return-right clear time            11 s
Overtake assist                    ON
Eval action noise                  OFF
Continuous action                  OFF
Debug evaluation                   ON
Video FPS                          30
```

如果仍发生普通车道追尾，优先依次调整：

```text
Normal-lane speed margin: 0.20 → 0.10 → 0.00
Speed-governor gap multiplier: 2.05 → 2.20 → 2.35
```

如果车辆过于保守、不愿超车：

```text
Lane-change min benefit: 26 → 22
Fast-lane safety multiplier: 1.90 → 1.75
```

## 公平比较不同版本

进行 V2、V3、V4、V5 对比时，应保持以下场景参数一致：

- 相同 seed
- 相同 traffic mode
- 相同天气
- 相同 episode duration
- Evaluation action noise 关闭
- 每个版本使用其对应版本训练出的模型

不能使用 V5 后端加载 V2 模型后再将结果称为 V2，因为这样会额外套用 V5 的 governor 和换道过滤规则。

详细说明见 [`docs/EVALUATION.md`](docs/EVALUATION.md)。

## 运行日志

训练日志和 TensorBoard 文件保存在 `runs/`：

```bash
tensorboard --logdir runs
```

Evaluation 开启 `--debug-eval` 后会输出：

- PPO 原始动作次数
- 实际执行动作次数
- Overtake assist 介入原因
- Safety shield 介入原因
- Speed governor 介入原因
- 被阻止的换道原因
- 车辆所在车道统计

## 模型与大文件

`.gitignore` 默认忽略：

- `models/*.zip`
- `videos/*.mp4`
- `comparison_videos/*.mp4`
- `runs/`

训练模型和视频通常不适合直接提交普通 Git。建议：

- 将最佳模型上传到 GitHub Release；或
- 使用 Git LFS；或
- 在 README 中提供云盘/Release 下载链接。

## 已知限制

- Safety layer 会修改部分 PPO 动作，因此效果不应描述为纯 PPO
- 前车识别主要基于车道索引，对正在横向切入的车辆仍可能存在短暂检测延迟
- 长时间 episode 的碰撞概率会累积增加
- 不同版本即使使用相同 seed，后续交通轨迹也可能因自车动作不同而逐渐分叉
- Continuous action 是实验功能，必须单独训练相应模型

## 将项目上传到 GitHub

在项目目录中运行：

```bash
git init
git add .
git commit -m "Release Highway PPO V5"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

如果仓库已经存在：

```bash
git add .
git commit -m "Update final V5 PPO implementation and documentation"
git push
```

## License

当前仓库未预设开源许可证。公开发布前，请根据课程或团队要求选择 MIT、Apache-2.0 或其他许可证。
