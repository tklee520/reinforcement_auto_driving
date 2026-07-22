# Highway PPO V5: Safe Overtaking in Long‑Tail Traffic

A highway driving project built on `highway-env` and Stable-Baselines3 PPO.  
The final V5 version combines a learned PPO policy with a safety layer and purposeful lane‑change logic, aiming to reduce rear‑end collisions, pointless lane changes, long‑term fast‑lane camping, and instability in complex traffic.

## Project highlights

- PPO outputs discrete driving actions: `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER`.
- 3‑second lane‑change cooldown to prevent rapid left‑right weaving.
- Weather‑aware safe distance and TTC thresholds.
- Risk checks on both the current lane and the target lane before lane changes.
- A speed governor that directly limits `ego.target_speed` so the car does not keep closing in at high speed while “IDLE”.
- Left‑lane overtakes are only allowed when there is a clear benefit.
- After overtaking, the car returns to the right as soon as the right lane is safe and useful.
- Supports `simple`, `realistic`, and `dense` traffic modes.
- Supports training, evaluation‑only runs, video recording, model selection, and Streamlit‑based parameter control.

## Project scope

This is **not** a “pure PPO with no rules”. The final system is intentionally hybrid:

```text
PPO policy
    ↓
Overtake / lane-purpose filtering
    ↓
Safety shield + speed governor
    ↓
Highway environment
```

PPO is responsible for proposing actions. The rule‑based layer filters obviously dangerous or pointless actions and limits target speed when rear‑end risk is high. In reports or presentations, we usually describe it like this:

> We train a PPO policy for highway driving and combine it with a safety assistance layer based on following distance, TTC, target‑lane state, and overtaking benefit, to improve safety and stability in long‑tail traffic scenarios.

## Repository structure

```text
.
├── train_highway_ppo_v5.py       # Backend for PPO training, evaluation, and video recording
├── v5.app.py                     # Streamlit dashboard for the V5 system
├── requirements.txt              # Python dependencies
├── MODEL_CARD.md                 # Model description and experiment notes template
├── CHANGELOG.md                  # Version history
├── docs/
│   ├── ARCHITECTURE.md           # System architecture
│   └── EVALUATION.md             # Fair comparison and video recording guidelines
├── scripts/
│   ├── run_dashboard.sh          # Start Streamlit dashboard
│   └── evaluate_example.sh       # Example evaluation script
├── models/                       # Trained models (not committed to Git by default)
├── videos/                       # Videos generated via Streamlit (not committed)
├── comparison_videos/            # Comparison videos across versions (not committed)
└── runs/                         # TensorBoard and evaluation logs
```

## Environment

Recommended setup:

- Python 3.10 or 3.11
- macOS, Linux, or Windows
- CPU is sufficient; CUDA GPU is useful for faster training

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start: Streamlit dashboard

The easiest way to explore the project is via Streamlit:

```bash
streamlit run v5.app.py
```

The dashboard provides four main pages:

1. Train and record videos
2. Load an existing `.zip` model and run evaluation only
3. Browse the video library
4. Parameter recommendations and notes

## Training from scratch

For a basic training run:

```bash
python train_highway_ppo_v5.py \
  --timesteps 300000 \
  --traffic-mode realistic \
  --duration 200 \
  --progress-bar
```

By default, models are saved under:

```text
models/
├── best/best_model.zip
├── checkpoints/
└── ppo_highway_longtail_realistic_v5_overtake_logic_discrete_realistic.zip
```

We recommend evaluating `models/best/best_model.zip` first. The final model saved at the end of training is not always the best checkpoint in terms of evaluation performance.

## Evaluation‑only with an existing model

To skip training and only evaluate a trained model:

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

Videos are written to:

```text
comparison_videos/v5_seed42/
```

## Video length and timing

`duration` is the **maximum** simulation time per episode, not the guaranteed MP4 playback length. Collisions or off‑road events can terminate an episode earlier.

In this project `policy_frequency = 5`, and `RecordVideo` generally records one frame per environment step. A rough approximation for video length is:

```text
video_seconds ≈ duration × 5 ÷ video_fps
```

Examples:

| duration | video_fps | approx. video length |
|--------:|----------:|---------------------:|
| 720     | 30        | ~120 s               |
| 600     | 10        | ~300 s               |
| 300     | 5         | ~300 s               |

Changing `video_fps` only affects MP4 playback speed. It does **not** change PPO decision frequency or driving behavior.

## Recommended demo settings

Below is a conservative but still overtaking‑friendly configuration we use for final demos:

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

If rear‑end collisions still appear in normal lanes, we usually adjust in this order:

```text
Normal-lane speed margin: 0.20 → 0.10 → 0.00
Speed-governor gap multiplier: 2.05 → 2.20 → 2.35
```

If the car becomes too conservative and rarely overtakes:

```text
Lane-change min benefit: 26 → 22
Fast-lane safety multiplier: 1.90 → 1.75
```

## Fair comparison across versions

When comparing V2, V3, V4, and V5, keep the scenario settings identical:

- Same `seed`
- Same `traffic-mode`
- Same weather
- Same `duration`
- Evaluation action noise **off**
- Each version uses a model trained with its own code base

It is **not** valid to load a V2 model in the V5 backend and call the result “V2”, because that would automatically apply the V5 speed governor and lane‑change filters. For more details, see [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Logs and TensorBoard

Training and evaluation logs are stored under `runs/`:

```bash
tensorboard --logdir runs
```

With `--debug-eval` enabled, the evaluation script additionally prints:

- Count of raw PPO actions
- Count of executed actions
- Reasons for overtake assist interventions
- Reasons for safety shield interventions
- Reasons for speed governor interventions
- Reasons for blocked lane changes
- Lane occupancy statistics

## Models and large files

The default `.gitignore` excludes:

- `models/*.zip`
- `videos/*.mp4`
- `comparison_videos/*.mp4`
- `runs/`

Trained models and videos are usually not suitable for normal Git commits. Typical options:

- Upload the best model to a GitHub Release.
- Use Git LFS for large files.
- Provide external download links in the README or in the model card.

## Known limitations

- The safety layer can modify PPO’s raw actions, so the system should not be described as “pure PPO”.
- Front‑vehicle detection mainly relies on lane indices; vehicles cutting in laterally can still cause short detection delays.
- Collision probability accumulates over very long episodes.
- Even with the same seed, different versions can diverge over time because the ego vehicle behaves differently.
- Continuous‑action support is experimental and requires separately trained models.
