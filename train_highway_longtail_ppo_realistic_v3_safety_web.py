#!/usr/bin/env python3
"""
Train/evaluate PPO on highway-v0 with realistic but active highway driving.

This V3 safety version keeps the active overtaking behaviour from V2, but adds
an extra safety shield for the overtaking/fast lane. This directly addresses the
common video failure mode: the ego car enters the fast lane, accelerates, and
then rear-ends a slower vehicle already in that lane.

Key ideas:
1. Keep the realistic constraints:
   - 3 second lane-change cooldown
   - no repeated lane changes while the current lane change is still happening
   - rain increases safe following distance
   - left-most fast lane is used mainly for overtaking, then the car returns right
2. Add an "overtake assist" layer:
   - if a slower vehicle is ahead and the left lane is safe, the wrapper can
     convert IDLE/FASTER into LANE_LEFT
   - if the car is in the left-most fast lane and the right lane is safe after
     overtaking, the wrapper can convert IDLE/FASTER into LANE_RIGHT
   - if the front gap is unsafe and overtaking is impossible, the wrapper slows down
3. Make the reward less timid:
   - lane changes are not punished when they are useful for overtaking
   - completing an overtake and leaving the fast lane are rewarded
   - smoothness is rewarded, but not so strongly that the agent refuses to steer
4. Make traffic more interesting but still trainable:
   - mixed speeds
   - some slow trucks
   - optional dense traffic mode
5. Add a fast-lane safety shield:
   - the fast/overtaking lane uses a larger following distance
   - acceleration is blocked in the fast lane when the front gap/TTC is unsafe
   - emergency slow-down overrides the PPO action if rear-end risk is high

Recommended commands:
    # Train the safer but active discrete version
    python train_highway_longtail_ppo_realistic_v3_safety_web.py --timesteps 300000 --duration 80

    # Evaluate and record videos
    python train_highway_longtail_ppo_realistic_v3_safety_web.py --skip-train --video-dir videos --duration 80 --debug-eval

    # Make traffic harder/more complex
    python train_highway_longtail_ppo_realistic_v3_safety_web.py --traffic-mode dense --timesteps 400000 --duration 80

    # Experimental continuous steering/throttle; this needs a fresh model
    python train_highway_longtail_ppo_realistic_v3_safety_web.py --continuous-action --timesteps 500000 --duration 80
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch as th
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

try:
    gym.register_envs(highway_env)
except Exception:
    pass

# DiscreteMetaAction action IDs used by highway-env.
LANE_LEFT = 0
IDLE = 1
LANE_RIGHT = 2
FASTER = 3
SLOWER = 4

ACTION_NAMES = {
    LANE_LEFT: "LANE_LEFT",
    IDLE: "IDLE",
    LANE_RIGHT: "LANE_RIGHT",
    FASTER: "FASTER",
    SLOWER: "SLOWER",
}


class RealisticHighwayWrapper(gym.Wrapper):
    """
    Highway wrapper with active, realistic driving behaviour.

    Discrete mode is recommended for your current project because it makes it
    easy to enforce the 3-second lane-change cooldown and overtaking-lane rules.
    Continuous mode is kept as an experimental option for smoother steering.
    """

    def __init__(
        self,
        env: gym.Env,
        seed: int,
        weather_intensity: Optional[float] = None,
        truck_stride: int = 3,
        train_mode: bool = True,
        enable_action_noise: bool = True,
        enable_overtake_assist: bool = True,
        lane_change_cooldown_sec: float = 3.0,
        fast_lane_safety_multiplier: float = 1.55,
        fast_lane_min_ttc: float = 3.2,
        fast_lane_accel_gap_multiplier: float = 1.35,
    ):
        super().__init__(env)

        self.rng = np.random.default_rng(seed)
        self.fixed_weather_intensity = weather_intensity
        self.truck_stride = max(1, int(truck_stride))
        self.train_mode = bool(train_mode)
        self.enable_action_noise = bool(enable_action_noise)
        self.enable_overtake_assist = bool(enable_overtake_assist)
        self.fast_lane_safety_multiplier = float(fast_lane_safety_multiplier)
        self.fast_lane_min_ttc = float(fast_lane_min_ttc)
        self.fast_lane_accel_gap_multiplier = float(fast_lane_accel_gap_multiplier)

        self.is_continuous_action = isinstance(self.env.action_space, spaces.Box)

        # Weather and actuation.
        self.weather_intensity = 0.5
        self.action_noise_prob = 0.0
        self.speed_reward_scale = 1.0

        # Lane-change realism.
        self.lane_change_cooldown_sec = float(lane_change_cooldown_sec)
        self.last_lane_change_time = -999.0
        self.last_lane_id: Optional[int] = None
        self.last_lane_change_direction: Optional[int] = None
        self.lane_change_block_reason = ""

        # Time/state tracking.
        self.simulation_time = 0.0
        self.fast_lane_time = 0.0
        self.previous_heading = 0.0
        self.previous_lateral_position = 0.0
        self.previous_continuous_action: Optional[np.ndarray] = None

        # Overtake tracking for rewards/debugging.
        self.overtake_active = False
        self.overtake_start_time = 0.0
        self.overtake_start_lane: Optional[int] = None
        self.overtake_reward_paid = False
        self.last_raw_action: Any = IDLE
        self.last_effective_action: Any = IDLE
        self.assist_reason = ""
        self.safety_shield_reason = ""
        self.current_front_gap: Optional[float] = None
        self.current_front_ttc: Optional[float] = None
        self.current_lane_safe_distance: Optional[float] = None

        # Distance and overtaking parameters.
        self.standstill_safe_distance = 10.0
        self.rain_extra_standstill_distance = 14.0
        self.dry_time_headway = 0.85
        self.rain_extra_time_headway = 0.95
        self.overtake_trigger_distance = 70.0
        self.overtake_clear_distance = 38.0
        self.min_speed_advantage_for_overtake = 1.5
        self.fast_lane_grace_sec = 4.0
        self.fast_lane_penalty = 0.05
        self.fast_lane_critical_gap_multiplier = 0.62
        self.fast_lane_caution_ttc_margin = 1.0
        self.normal_lane_min_ttc = 2.0
        self.rain_extra_ttc = 1.4

        # Continuous action smoothing parameters.
        self.continuous_action_alpha = 0.45
        self.max_continuous_action_delta = 0.22
        self.cooldown_steering_scale = 0.55

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)

        self.weather_intensity = (
            float(self.fixed_weather_intensity)
            if self.fixed_weather_intensity is not None
            else float(self.rng.uniform(0.15, 0.75))
        )

        # Keep noise small. Evaluation noise is off unless explicitly requested.
        if self.train_mode and self.enable_action_noise:
            self.action_noise_prob = 0.01 + 0.05 * self.weather_intensity
        else:
            self.action_noise_prob = 0.0

        # Rain still lowers the target a bit, but not enough to make the car passive.
        self.speed_reward_scale = 1.0 - 0.15 * self.weather_intensity

        self.simulation_time = 0.0
        self.last_lane_change_time = -999.0
        self.fast_lane_time = 0.0
        self.lane_change_block_reason = ""
        self.assist_reason = ""
        self.safety_shield_reason = ""
        self.current_front_gap = None
        self.current_front_ttc = None
        self.current_lane_safe_distance = None
        self.last_lane_change_direction = None
        self.overtake_active = False
        self.overtake_start_time = 0.0
        self.overtake_start_lane = None
        self.overtake_reward_paid = False
        self.last_raw_action = IDLE
        self.last_effective_action = IDLE

        ego = self.env.unwrapped.vehicle
        self.last_lane_id = self._lane_id(ego)
        self.previous_heading = float(getattr(ego, "heading", 0.0))
        self.previous_lateral_position = float(ego.position[1])

        if self.is_continuous_action:
            self.previous_continuous_action = np.zeros(
                self.env.action_space.shape,
                dtype=np.float32,
            )
        else:
            self.previous_continuous_action = None

        self._make_traffic_mixed_and_interesting()

        info = dict(info)
        info.update(self._common_info(IDLE))
        return obs, info

    def step(self, action: Any):
        self.simulation_time += self._policy_dt()
        self.lane_change_block_reason = ""
        self.assist_reason = ""
        self.safety_shield_reason = ""
        self.current_front_gap = None
        self.current_front_ttc = None
        self.current_lane_safe_distance = None

        ego_before = self.env.unwrapped.vehicle
        self._update_lane_tracking(ego_before)
        self._update_overtake_state_before_action(ego_before)

        self.last_raw_action = self._action_to_info_value(action)
        filtered_action = self._filter_action(action)
        self.last_effective_action = self._action_to_info_value(filtered_action)

        obs, _, terminated, truncated, info = self.env.step(filtered_action)

        ego_after = self.env.unwrapped.vehicle
        self._update_lane_tracking(ego_after)
        self._update_overtake_state_after_action(ego_after)

        reward = self._compute_reward(filtered_action)

        info = dict(info)
        info.update(self._common_info(filtered_action))
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action filtering / driving assist
    # ------------------------------------------------------------------
    def _filter_action(self, action: Any):
        if self.is_continuous_action:
            return self._smooth_continuous_action(action)

        discrete_action = int(np.asarray(action).item())
        discrete_action = self._apply_action_noise(discrete_action)
        discrete_action = self._apply_overtake_assist(discrete_action)
        discrete_action = self._apply_discrete_lane_rules(discrete_action)
        discrete_action = self._apply_discrete_safety_shield(discrete_action)
        return int(discrete_action)

    def _apply_overtake_assist(self, action: int) -> int:
        """Make the agent visibly drive and overtake without becoming reckless.

        PPO sometimes learns that the safest policy is to keep IDLE/SLOWER. This
        assistant is similar to a rule-based ADAS layer: it only changes actions
        when there is a clear traffic reason and a safe gap.
        """
        if not self.enable_overtake_assist:
            return action

        ego = self.env.unwrapped.vehicle

        # Safety has priority over making the car look active. If the front gap is
        # already critical, slow first instead of trying a late lane change.
        critical, caution, reason = self._front_safety_state(ego)
        if critical and action != SLOWER:
            self.assist_reason = "critical_front_slow_down"
            return SLOWER
        if caution and action == FASTER:
            self.assist_reason = "front_gap_block_speed_up"
            return IDLE

        # Never force new lane changes while a lane change or cooldown is active.
        if self._lane_change_in_progress(ego) or not self._cooldown_finished():
            if self._following_too_close(ego) and action not in (SLOWER, LANE_LEFT, LANE_RIGHT):
                self.assist_reason = "too_close_slow_down"
                return SLOWER
            return action

        # Leave the left-most fast/overtaking lane as soon as the pass is done.
        if self._should_leave_fast_lane(ego) and self._can_change_lane(ego, LANE_RIGHT):
            if action in (IDLE, FASTER, SLOWER, LANE_LEFT):
                self.assist_reason = "return_right_after_overtake"
                return LANE_RIGHT

        # Start overtaking earlier: do not wait until ego is already faster than the front car.
        if self._overtake_opportunity(ego):
            if self._can_change_lane(ego, LANE_LEFT):
                if action in (IDLE, FASTER, SLOWER):
                    self.assist_reason = "start_overtake_left"
                    return LANE_LEFT
            elif self._following_too_close(ego) and action != SLOWER:
                self.assist_reason = "too_close_no_gap_slow_down"
                return SLOWER

        # If road ahead is open, prefer maintaining speed instead of being passive.
        if action == IDLE and not self._following_too_close(ego):
            speed = float(getattr(ego, "speed", 0.0))
            critical, caution, _ = self._front_safety_state(ego)
            if not critical and not caution and speed < self._target_cruise_speed() - 3.0:
                self.assist_reason = "open_road_speed_up"
                return FASTER

        return action

    def _apply_discrete_lane_rules(self, action: int) -> int:
        ego = self.env.unwrapped.vehicle

        if action in (LANE_LEFT, LANE_RIGHT):
            if self._lane_change_in_progress(ego):
                self.lane_change_block_reason = "lane_change_in_progress"
                return IDLE

            if not self._cooldown_finished():
                self.lane_change_block_reason = "cooldown"
                return IDLE

            if not self._can_change_lane(ego, action):
                self.lane_change_block_reason = "unsafe_or_out_of_road"
                return IDLE

            target_lane_id = self._target_lane_id_for_action(ego, action)

            # Do not enter the left-most fast lane without an overtake reason.
            if target_lane_id == 0 and not self._overtake_opportunity(ego) and not self._is_overtaking(ego):
                self.lane_change_block_reason = "no_overtake_need_for_fast_lane"
                return IDLE

            self.last_lane_change_time = self.simulation_time
            self.last_lane_change_direction = action
            return action

        # Even if the policy does nothing, do not camp in the fast lane.
        if action in (IDLE, FASTER, SLOWER) and self._should_leave_fast_lane(ego):
            if self._cooldown_finished() and not self._lane_change_in_progress(ego) and self._can_change_lane(ego, LANE_RIGHT):
                self.last_lane_change_time = self.simulation_time
                self.last_lane_change_direction = LANE_RIGHT
                self.assist_reason = self.assist_reason or "forced_return_right"
                return LANE_RIGHT

        return action

    def _apply_discrete_safety_shield(self, action: int) -> int:
        """Hard safety layer for rear-end prevention.

        Reward penalties alone are not enough: PPO can still choose FASTER in a
        dangerous gap, especially after moving into the fast/overtaking lane.
        This shield overrides only the most dangerous actions.
        """
        ego = self.env.unwrapped.vehicle
        lane_id = self._lane_id(ego)

        critical, caution, reason = self._front_safety_state(ego)

        # If the front gap/TTC is critical, braking is safer than accelerating or
        # gambling on a late lane change. This is especially important in the
        # fast lane where closing speeds are higher.
        if critical and action != SLOWER:
            self.safety_shield_reason = f"critical_front_{reason}"
            return SLOWER

        # If the situation is not yet critical but is already unsafe, block
        # acceleration. In the fast lane, be even more conservative.
        if action == FASTER and caution:
            self.safety_shield_reason = f"block_accel_{reason}"
            return IDLE

        if lane_id == 0 and action == FASTER:
            front_vehicle, front_gap = self._front_vehicle_and_gap(ego)
            if front_vehicle is not None and front_gap is not None:
                speed = float(getattr(ego, "speed", 0.0))
                fast_safe_distance = self._lane_safe_following_distance(speed, lane_id)
                ttc = self._time_to_collision(ego, front_vehicle, front_gap)
                min_ttc = self._lane_min_ttc(lane_id)

                # Acceleration in the overtaking lane is allowed only when the
                # front gap is comfortably larger than the required safety gap.
                gap_not_clear = front_gap < self.fast_lane_accel_gap_multiplier * fast_safe_distance
                ttc_not_clear = ttc is not None and ttc < min_ttc + self.fast_lane_caution_ttc_margin

                if gap_not_clear or ttc_not_clear:
                    self.safety_shield_reason = "fast_lane_accel_limited"
                    return IDLE

        return action

    def _apply_action_noise(self, action: int) -> int:
        if self.action_noise_prob <= 0.0:
            return action
        if self.rng.random() >= self.action_noise_prob:
            return action
        # Noise only delays an action; it does not reverse intention.
        if action in (LANE_LEFT, LANE_RIGHT, FASTER, SLOWER):
            return IDLE
        return action

    def _smooth_continuous_action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).copy()
        arr = np.reshape(arr, self.env.action_space.shape)
        arr = np.clip(arr, self.env.action_space.low, self.env.action_space.high)

        if self.previous_continuous_action is None:
            self.previous_continuous_action = np.zeros_like(arr, dtype=np.float32)

        prev = self.previous_continuous_action
        smoothed = self.continuous_action_alpha * arr + (1.0 - self.continuous_action_alpha) * prev
        smoothed = np.clip(
            smoothed,
            prev - self.max_continuous_action_delta,
            prev + self.max_continuous_action_delta,
        )

        if smoothed.size >= 2:
            flat = smoothed.reshape(-1)
            steering_index = 1
            if not self._cooldown_finished():
                flat[steering_index] *= self.cooldown_steering_scale
            flat[steering_index] *= 1.0 - 0.15 * self.weather_intensity
            smoothed = flat.reshape(smoothed.shape)

        smoothed = self._apply_continuous_safety_shield(smoothed)
        smoothed = np.clip(smoothed, self.env.action_space.low, self.env.action_space.high)
        self.previous_continuous_action = smoothed.copy()
        return smoothed.astype(np.float32)

    def _apply_continuous_safety_shield(self, action: np.ndarray) -> np.ndarray:
        """Limit acceleration/brake continuously when front risk is high."""
        if action.size < 1:
            return action

        ego = self.env.unwrapped.vehicle
        lane_id = self._lane_id(ego)
        critical, caution, reason = self._front_safety_state(ego)

        flat = action.reshape(-1).copy()

        # In highway-env ContinuousAction, positive longitudinal action means
        # acceleration and negative means braking/deceleration in normal configs.
        if critical:
            flat[0] = min(float(flat[0]), -0.65)
            self.safety_shield_reason = f"continuous_critical_front_{reason}"
        elif caution:
            flat[0] = min(float(flat[0]), 0.0)
            self.safety_shield_reason = f"continuous_block_accel_{reason}"

        if lane_id == 0:
            front_vehicle, front_gap = self._front_vehicle_and_gap(ego)
            if front_vehicle is not None and front_gap is not None:
                speed = float(getattr(ego, "speed", 0.0))
                fast_safe_distance = self._lane_safe_following_distance(speed, lane_id)
                ttc = self._time_to_collision(ego, front_vehicle, front_gap)
                if front_gap < self.fast_lane_accel_gap_multiplier * fast_safe_distance:
                    flat[0] = min(float(flat[0]), 0.0)
                    self.safety_shield_reason = self.safety_shield_reason or "continuous_fast_lane_accel_limited"
                if ttc is not None and ttc < self._lane_min_ttc(lane_id):
                    flat[0] = min(float(flat[0]), -0.35)
                    self.safety_shield_reason = self.safety_shield_reason or "continuous_fast_lane_ttc_brake"

        return flat.reshape(action.shape).astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, action: Any) -> float:
        ego = self.env.unwrapped.vehicle
        cfg = self.env.unwrapped.config

        if getattr(ego, "crashed", False):
            return float(cfg.get("collision_reward", -100.0))

        reward = 0.0
        speed = float(getattr(ego, "speed", 0.0))
        heading = float(getattr(ego, "heading", 0.0))
        forward_speed = max(0.0, speed * float(np.cos(heading)))

        target_speed = self._target_cruise_speed()
        speed_score = np.clip(forward_speed / max(target_speed, 1.0), 0.0, 1.2)
        reward += 4.5 * min(speed_score, 1.0) * self.speed_reward_scale

        # Do not learn a passive stopped policy.
        min_useful_speed = 10.0
        if forward_speed < min_useful_speed:
            reward -= 2.0 * (1.0 - forward_speed / min_useful_speed)

        front_vehicle, front_gap = self._front_vehicle_and_gap(ego)
        if front_vehicle is not None and front_gap is not None:
            safe_distance = self._lane_safe_following_distance(speed, self._lane_id(ego))
            front_speed = float(getattr(front_vehicle, "speed", 0.0))
            closing_speed = max(0.0, forward_speed - front_speed)

            if front_gap < safe_distance:
                distance_ratio = max(front_gap / max(safe_distance, 1e-6), 0.0)
                # Still meaningful, but less paralyzing than V1.
                reward -= 3.2 * (1.0 - distance_ratio) ** 2 * (1.0 + self.weather_intensity)

            if closing_speed > 0.2:
                ttc = front_gap / closing_speed
                min_ttc = self._lane_min_ttc(self._lane_id(ego))
                if ttc < min_ttc:
                    reward -= 3.4 * (1.0 - ttc / min_ttc) * (1.0 + self.weather_intensity)

        if not self.is_continuous_action:
            discrete_action = int(action)
            if discrete_action == LANE_LEFT:
                if self._overtake_opportunity(ego) or self.overtake_active:
                    reward += 0.65
                else:
                    reward -= 0.12 * (1.0 + self.weather_intensity)
            elif discrete_action == LANE_RIGHT:
                if self._should_leave_fast_lane(ego) or self.overtake_active:
                    reward += 0.45
                else:
                    reward -= 0.06
            elif discrete_action == FASTER and not self._following_too_close(ego):
                if self._lane_id(ego) == 0:
                    _, gap = self._front_vehicle_and_gap(ego)
                    if gap is not None and gap < self.fast_lane_accel_gap_multiplier * self._lane_safe_following_distance(speed, 0):
                        reward -= 0.45
                    else:
                        reward += 0.04
                else:
                    reward += 0.08
            elif discrete_action == SLOWER and self._following_too_close(ego):
                reward += 0.12

        lane_id = self._lane_id(ego)
        max_lane_id = self._max_lane_id(ego)
        if lane_id is not None and max_lane_id is not None and max_lane_id > 0:
            # Larger lane IDs are the right-most lanes in highway-env.
            right_lane_bonus = lane_id / max_lane_id
            reward += 0.10 * right_lane_bonus

            if lane_id == 0:
                self.fast_lane_time += self._policy_dt()
                _, fast_front_gap = self._front_vehicle_and_gap(ego)
                if fast_front_gap is not None:
                    fast_safe = self._lane_safe_following_distance(speed, lane_id)
                    if fast_front_gap < fast_safe:
                        reward -= 0.55 * (1.0 - fast_front_gap / max(fast_safe, 1e-6))
                if self._is_overtaking(ego):
                    reward += 0.25
                elif self.fast_lane_time > self.fast_lane_grace_sec:
                    reward -= self.fast_lane_penalty + 0.012 * (self.fast_lane_time - self.fast_lane_grace_sec)
                    if self._can_change_lane(ego, LANE_RIGHT):
                        reward -= 0.10
            else:
                # Paying a small bonus after leaving the fast lane prevents camping.
                if self.overtake_reward_paid:
                    reward += 0.35
                    self.overtake_reward_paid = False
                self.fast_lane_time = 0.0

        lateral_position = float(ego.position[1])
        heading_change = abs(self._angle_difference(heading, self.previous_heading))
        lateral_change = abs(lateral_position - self.previous_lateral_position)

        # Reduced vs V1: smooth driving matters, but should not suppress all steering.
        reward -= 0.20 * heading_change + 0.035 * lateral_change

        if self.is_continuous_action and self.previous_continuous_action is not None:
            act = self.previous_continuous_action.reshape(-1)
            reward -= 0.012 * float(np.linalg.norm(act, ord=2))
            if act.size >= 2:
                reward -= 0.02 * abs(float(act[1]))

        self.previous_heading = heading
        self.previous_lateral_position = lateral_position

        if not bool(getattr(ego, "on_road", True)):
            reward -= 25.0

        return float(reward)

    # ------------------------------------------------------------------
    # Lane, gap and overtaking helpers
    # ------------------------------------------------------------------
    def _policy_dt(self) -> float:
        policy_frequency = float(self.env.unwrapped.config.get("policy_frequency", 5))
        return 1.0 / max(policy_frequency, 1.0)

    def _cooldown_finished(self) -> bool:
        return (self.simulation_time - self.last_lane_change_time) >= self.lane_change_cooldown_sec

    def _lane_id(self, vehicle: Any) -> Optional[int]:
        if vehicle is None:
            return None
        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is None:
            return None
        try:
            return int(lane_index[2])
        except Exception:
            return None

    def _max_lane_id(self, vehicle: Any) -> Optional[int]:
        try:
            lanes = self.env.unwrapped.road.network.all_side_lanes(vehicle.lane_index)
            return max(len(lanes) - 1, 0)
        except Exception:
            lanes_count = self.env.unwrapped.config.get("lanes_count", None)
            if lanes_count is None:
                return None
            return int(lanes_count) - 1

    def _target_lane_id_for_action(self, ego: Any, action: int) -> Optional[int]:
        lane_id = self._lane_id(ego)
        if lane_id is None:
            return None
        if action == LANE_LEFT:
            return lane_id - 1
        if action == LANE_RIGHT:
            return lane_id + 1
        return lane_id

    def _lane_change_in_progress(self, ego: Any) -> bool:
        lane_index = getattr(ego, "lane_index", None)
        target_lane_index = getattr(ego, "target_lane_index", None)
        if lane_index is None or target_lane_index is None:
            return False
        try:
            return int(lane_index[2]) != int(target_lane_index[2])
        except Exception:
            return lane_index != target_lane_index

    def _update_lane_tracking(self, ego: Any) -> None:
        lane_id = self._lane_id(ego)
        if lane_id is None:
            return
        if self.last_lane_id is None:
            self.last_lane_id = lane_id
            return
        if lane_id != self.last_lane_id:
            self.last_lane_change_time = self.simulation_time
            self.last_lane_id = lane_id

    def _can_change_lane(self, ego: Any, action: int) -> bool:
        target_lane_id = self._target_lane_id_for_action(ego, action)
        max_lane_id = self._max_lane_id(ego)
        if target_lane_id is None or max_lane_id is None:
            return False
        if target_lane_id < 0 or target_lane_id > max_lane_id:
            return False
        return self._lane_gap_safe(ego, target_lane_id)

    def _lane_gap_safe(self, ego: Any, target_lane_id: int) -> bool:
        road = getattr(self.env.unwrapped, "road", None)
        if road is None:
            return True

        ego_s = self._longitudinal_position_on_lane(ego, target_lane_id)
        if ego_s is None:
            return True

        ego_speed = float(getattr(ego, "speed", 0.0))
        target_safe_distance = self._lane_safe_following_distance(ego_speed, target_lane_id)

        if target_lane_id == 0:
            # Entering the fast/overtaking lane is riskier because vehicles there
            # can be faster and closing speeds are larger. Require a larger gap.
            front_min_gap = max(18.0, 0.95 * target_safe_distance)
            rear_min_gap = 14.0 + 12.0 * self.weather_intensity
        else:
            front_min_gap = max(11.0, 0.65 * target_safe_distance)
            rear_min_gap = 9.0 + 9.0 * self.weather_intensity

        front_min_ttc = self._lane_min_ttc(target_lane_id)
        rear_min_ttc = self._lane_min_ttc(target_lane_id) + (0.8 if target_lane_id == 0 else 0.2)

        for other in road.vehicles:
            if other is ego:
                continue
            if self._lane_id(other) != target_lane_id:
                continue

            other_s = self._longitudinal_position_on_lane(other, target_lane_id)
            if other_s is None:
                continue

            delta = other_s - ego_s
            other_length = float(getattr(other, "LENGTH", 5.0))
            ego_length = float(getattr(ego, "LENGTH", 5.0))
            bumper_gap = abs(delta) - 0.5 * (other_length + ego_length)

            other_speed = float(getattr(other, "speed", 0.0))

            if delta >= 0.0:
                if bumper_gap < front_min_gap:
                    return False
                closing_speed = max(0.0, ego_speed - other_speed)
                if closing_speed > 0.2 and bumper_gap / closing_speed < front_min_ttc:
                    return False

            if delta < 0.0:
                if bumper_gap < rear_min_gap:
                    return False
                rear_closing_speed = max(0.0, other_speed - ego_speed)
                if rear_closing_speed > 0.2 and bumper_gap / rear_closing_speed < rear_min_ttc:
                    return False

        return True

    def _front_vehicle_and_gap(self, ego: Any):
        lane_id = self._lane_id(ego)
        if lane_id is None:
            return None, None
        return self._closest_front_in_lane(ego, lane_id)

    def _closest_front_in_lane(self, ego: Any, lane_id: int):
        road = getattr(self.env.unwrapped, "road", None)
        if road is None:
            return None, None

        ego_s = self._longitudinal_position_on_lane(ego, lane_id)
        if ego_s is None:
            return None, None

        closest = None
        closest_gap = None
        ego_length = float(getattr(ego, "LENGTH", 5.0))

        for other in road.vehicles:
            if other is ego:
                continue
            if self._lane_id(other) != lane_id:
                continue
            other_s = self._longitudinal_position_on_lane(other, lane_id)
            if other_s is None:
                continue
            delta = other_s - ego_s
            if delta <= 0.0:
                continue
            gap = delta - 0.5 * (ego_length + float(getattr(other, "LENGTH", 5.0)))
            if closest_gap is None or gap < closest_gap:
                closest = other
                closest_gap = float(max(0.0, gap))

        return closest, closest_gap

    def _longitudinal_position_on_lane(self, vehicle: Any, lane_id: int) -> Optional[float]:
        road = getattr(self.env.unwrapped, "road", None)
        lane_index = getattr(vehicle, "lane_index", None)
        if road is None or lane_index is None:
            return None

        try:
            lane_index_for_projection = (lane_index[0], lane_index[1], lane_id)
            lane = road.network.get_lane(lane_index_for_projection)
            longitudinal, _ = lane.local_coordinates(vehicle.position)
            return float(longitudinal)
        except Exception:
            try:
                return float(vehicle.position[0])
            except Exception:
                return None

    def _target_cruise_speed(self) -> float:
        speed_range = self.env.unwrapped.config.get("reward_speed_range", [20.0, 30.0])
        dry_target = float(speed_range[1])
        rainy_target = dry_target - 3.0 * self.weather_intensity
        return max(20.0, rainy_target)

    def _safe_following_distance(self, speed: float) -> float:
        standstill = self.standstill_safe_distance + self.rain_extra_standstill_distance * self.weather_intensity
        time_headway = self.dry_time_headway + self.rain_extra_time_headway * self.weather_intensity
        return standstill + max(0.0, float(speed)) * time_headway

    def _lane_safe_following_distance(self, speed: float, lane_id: Optional[int]) -> float:
        """Return the desired following distance for the current lane.

        The left-most fast/overtaking lane deliberately uses a larger distance.
        This solves the common failure where the car moves left, accelerates, and
        then rear-ends a slower vehicle already in the fast lane.
        """
        base_distance = self._safe_following_distance(speed)
        if lane_id == 0:
            return base_distance * self.fast_lane_safety_multiplier
        return base_distance

    def _lane_min_ttc(self, lane_id: Optional[int]) -> float:
        if lane_id == 0:
            return self.fast_lane_min_ttc + self.rain_extra_ttc * self.weather_intensity
        return self.normal_lane_min_ttc + self.rain_extra_ttc * self.weather_intensity

    def _time_to_collision(self, ego: Any, front_vehicle: Any, front_gap: float) -> Optional[float]:
        ego_speed = float(getattr(ego, "speed", 0.0))
        front_speed = float(getattr(front_vehicle, "speed", 0.0))
        closing_speed = ego_speed - front_speed
        if closing_speed <= 0.2:
            return None
        return float(front_gap / closing_speed)

    def _front_safety_state(self, ego: Any, lane_id: Optional[int] = None) -> tuple[bool, bool, str]:
        """Return (critical, caution, reason) for front rear-end risk.

        critical -> override to SLOWER.
        caution  -> block acceleration.
        """
        if lane_id is None:
            lane_id = self._lane_id(ego)
        if lane_id is None:
            return False, False, "no_lane"

        front_vehicle, front_gap = self._closest_front_in_lane(ego, lane_id)
        if front_vehicle is None or front_gap is None:
            self.current_front_gap = None
            self.current_front_ttc = None
            self.current_lane_safe_distance = None
            return False, False, "no_front"

        speed = float(getattr(ego, "speed", 0.0))
        safe_distance = self._lane_safe_following_distance(speed, lane_id)
        min_ttc = self._lane_min_ttc(lane_id)
        ttc = self._time_to_collision(ego, front_vehicle, front_gap)

        self.current_front_gap = float(front_gap)
        self.current_front_ttc = ttc
        self.current_lane_safe_distance = float(safe_distance)

        if lane_id == 0:
            critical_gap_multiplier = self.fast_lane_critical_gap_multiplier
            caution_gap_multiplier = 1.0
        else:
            critical_gap_multiplier = 0.55
            caution_gap_multiplier = 0.85

        critical_gap = front_gap < critical_gap_multiplier * safe_distance
        caution_gap = front_gap < caution_gap_multiplier * safe_distance
        critical_ttc = ttc is not None and ttc < min_ttc
        caution_ttc = ttc is not None and ttc < min_ttc + (self.fast_lane_caution_ttc_margin if lane_id == 0 else 0.6)

        if critical_gap:
            return True, True, "gap"
        if critical_ttc:
            return True, True, "ttc"
        if caution_gap:
            return False, True, "gap"
        if caution_ttc:
            return False, True, "ttc"
        return False, False, "clear"

    def _following_too_close(self, ego: Any) -> bool:
        _, front_gap = self._front_vehicle_and_gap(ego)
        if front_gap is None:
            return False
        speed = float(getattr(ego, "speed", 0.0))
        lane_id = self._lane_id(ego)
        return front_gap < 0.85 * self._lane_safe_following_distance(speed, lane_id)

    def _overtake_opportunity(self, ego: Any) -> bool:
        """True when a slower front vehicle makes a left-lane pass useful."""
        lane_id = self._lane_id(ego)
        if lane_id is None or lane_id <= 0:
            return False

        front_vehicle, front_gap = self._front_vehicle_and_gap(ego)
        if front_vehicle is None or front_gap is None:
            return False

        ego_speed = float(getattr(ego, "speed", 0.0))
        front_speed = float(getattr(front_vehicle, "speed", 0.0))
        target_speed = self._target_cruise_speed()
        safe_distance = self._lane_safe_following_distance(ego_speed, lane_id)

        # If the car is already too close to the front vehicle, slow down first;
        # do not start a late, risky overtake as an emergency escape.
        critical, _, _ = self._front_safety_state(ego, lane_id)
        if critical:
            return False

        slow_front_car = front_speed < target_speed - self.min_speed_advantage_for_overtake
        close_enough = front_gap < max(self.overtake_trigger_distance, 1.35 * safe_distance)
        closing_or_will_be_blocked = ego_speed + 0.5 >= front_speed or front_gap < safe_distance

        return bool(slow_front_car and close_enough and closing_or_will_be_blocked)

    def _is_overtaking(self, ego: Any) -> bool:
        lane_id = self._lane_id(ego)
        max_lane_id = self._max_lane_id(ego)
        if lane_id is None or max_lane_id is None:
            return False
        if lane_id != 0:
            return False

        right_lane_id = lane_id + 1
        if right_lane_id > max_lane_id:
            return False

        ego_s = self._longitudinal_position_on_lane(ego, right_lane_id)
        if ego_s is None:
            return False

        ego_speed = float(getattr(ego, "speed", 0.0))
        road = getattr(self.env.unwrapped, "road", None)
        if road is None:
            return False

        for other in road.vehicles:
            if other is ego or self._lane_id(other) != right_lane_id:
                continue
            other_s = self._longitudinal_position_on_lane(other, right_lane_id)
            if other_s is None:
                continue
            delta = other_s - ego_s
            other_speed = float(getattr(other, "speed", 0.0))
            if -18.0 <= delta <= self.overtake_clear_distance and other_speed <= ego_speed + 2.0:
                return True
        return False

    def _should_leave_fast_lane(self, ego: Any) -> bool:
        lane_id = self._lane_id(ego)
        if lane_id != 0:
            return False
        if self._is_overtaking(ego):
            return False
        # Give a short grace period after entering so it can complete the pass.
        if self.fast_lane_time < self.fast_lane_grace_sec:
            return False
        return True

    def _update_overtake_state_before_action(self, ego: Any) -> None:
        if self.overtake_active:
            return
        if self._overtake_opportunity(ego):
            self.overtake_active = True
            self.overtake_start_time = self.simulation_time
            self.overtake_start_lane = self._lane_id(ego)
            self.overtake_reward_paid = False

    def _update_overtake_state_after_action(self, ego: Any) -> None:
        if not self.overtake_active:
            return
        lane_id = self._lane_id(ego)
        # End the overtake after the car has returned to a non-fast lane and
        # there is no immediate slow vehicle ahead.
        if lane_id is not None and lane_id > 0 and not self._overtake_opportunity(ego):
            if self.simulation_time - self.overtake_start_time > 1.0:
                self.overtake_active = False
                self.overtake_reward_paid = True

    @staticmethod
    def _angle_difference(a: float, b: float) -> float:
        return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)

    # ------------------------------------------------------------------
    # Traffic modification and info
    # ------------------------------------------------------------------
    def _make_traffic_mixed_and_interesting(self) -> None:
        road = getattr(self.env.unwrapped, "road", None)
        if road is None:
            return

        # Create a mix of slow trucks and normal vehicles. This makes overtaking
        # necessary without turning the environment into random chaos.
        for idx, vehicle in enumerate(road.vehicles[1:], start=1):
            lane_id = self._lane_id(vehicle)

            if idx % self.truck_stride == 0:
                vehicle.LENGTH = 8.5
                vehicle.WIDTH = 2.75
                vehicle.MAX_SPEED = min(float(getattr(vehicle, "MAX_SPEED", 30.0)), 23.0)
                slow_speed = float(self.rng.uniform(17.0, 22.0))
                if hasattr(vehicle, "target_speed"):
                    vehicle.target_speed = min(float(vehicle.target_speed), slow_speed)
                if hasattr(vehicle, "speed"):
                    vehicle.speed = min(float(vehicle.speed), slow_speed + 1.0)
            elif idx % 5 == 0:
                medium_speed = float(self.rng.uniform(20.0, 25.0))
                if hasattr(vehicle, "target_speed"):
                    vehicle.target_speed = min(float(vehicle.target_speed), medium_speed)
            else:
                # Keep some faster traffic too, especially on the left lanes.
                if hasattr(vehicle, "target_speed"):
                    base = 27.0 if lane_id in (0, 1) else 24.0
                    vehicle.target_speed = float(self.rng.uniform(base - 2.0, base + 2.0))

    def _action_to_info_value(self, action: Any) -> Any:
        if self.is_continuous_action:
            return np.asarray(action).round(3).tolist()
        try:
            value = int(np.asarray(action).item())
            return ACTION_NAMES.get(value, value)
        except Exception:
            return action

    def _common_info(self, effective_action: Any) -> dict[str, Any]:
        ego = getattr(self.env.unwrapped, "vehicle", None)
        lane_id = self._lane_id(ego) if ego is not None else None
        safe_distance = self._lane_safe_following_distance(float(getattr(ego, "speed", 0.0)), lane_id) if ego is not None else None

        return {
            "weather_intensity": float(self.weather_intensity),
            "action_noise_prob": float(self.action_noise_prob),
            "raw_action": self.last_raw_action,
            "effective_action": self._action_to_info_value(effective_action),
            "assist_reason": self.assist_reason,
            "safety_shield_reason": self.safety_shield_reason,
            "lane_change_block_reason": self.lane_change_block_reason,
            "speed_reward_scale": float(self.speed_reward_scale),
            "lane_change_cooldown_sec": float(self.lane_change_cooldown_sec),
            "safe_following_distance": safe_distance,
            "current_front_gap": self.current_front_gap,
            "current_front_ttc": self.current_front_ttc,
            "current_lane_safe_distance": self.current_lane_safe_distance,
            "lane_id": lane_id,
            "fast_lane_time": float(self.fast_lane_time),
            "overtake_active": bool(self.overtake_active),
            "overtake_opportunity": bool(self._overtake_opportunity(ego)) if ego is not None else False,
            "following_too_close": bool(self._following_too_close(ego)) if ego is not None else False,
            "continuous_action": bool(self.is_continuous_action),
        }


def build_env(
    seed: int,
    render_mode: Optional[str] = None,
    eval_weather: Optional[float] = None,
    train_mode: bool = True,
    continuous_action: bool = False,
    enable_action_noise: bool = True,
    enable_overtake_assist: bool = True,
    traffic_mode: str = "realistic",
    duration: int = 80,
    fast_lane_safety_multiplier: float = 1.55,
    fast_lane_min_ttc: float = 3.2,
    fast_lane_accel_gap_multiplier: float = 1.35,
):
    if continuous_action:
        action_config: dict[str, Any] = {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
            "acceleration_range": [-4.0, 3.0],
            "steering_range": [-0.35, 0.35],
        }
    else:
        action_config = {"type": "DiscreteMetaAction"}

    if traffic_mode == "simple":
        vehicles_count = 16
        vehicles_density = 0.75
        truck_stride = 4
    elif traffic_mode == "dense":
        vehicles_count = 28
        vehicles_density = 1.25
        truck_stride = 2
    else:
        vehicles_count = 22
        vehicles_density = 1.0
        truck_stride = 3

    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 12,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": True,
            "order": "sorted",
        },
        "action": action_config,
        "lanes_count": 4,
        "vehicles_count": vehicles_count,
        "controlled_vehicles": 1,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "collision_reward": -100.0,
        "high_speed_reward": 5.0,
        "right_lane_reward": 0.05,
        "lane_change_reward": 0.0,
        "reward_speed_range": [20.0, 31.0],
        "normalize_reward": False,
        "offroad_terminal": True,
        "duration": int(duration),
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "ego_spacing": 2,
        "vehicles_density": vehicles_density,
        "screen_width": 960,
        "screen_height": 260,
        "centering_position": [0.3, 0.5],
        "scaling": 5.0,
        "show_trajectories": False,
        "render_agent": True,
        "manual_control": False,
        "real_time_rendering": False,
    }

    env = gym.make("highway-v0", config=config, render_mode=render_mode)
    env = RealisticHighwayWrapper(
        env,
        seed=seed,
        weather_intensity=eval_weather,
        truck_stride=truck_stride,
        train_mode=train_mode,
        enable_action_noise=enable_action_noise,
        enable_overtake_assist=enable_overtake_assist,
        fast_lane_safety_multiplier=fast_lane_safety_multiplier,
        fast_lane_min_ttc=fast_lane_min_ttc,
        fast_lane_accel_gap_multiplier=fast_lane_accel_gap_multiplier,
    )
    env = Monitor(env)
    return env


def make_train_env(seed: int, args: argparse.Namespace):
    return build_env(
        seed=seed,
        render_mode=None,
        eval_weather=None,
        train_mode=True,
        continuous_action=args.continuous_action,
        enable_action_noise=not args.no_action_noise,
        enable_overtake_assist=not args.no_overtake_assist,
        traffic_mode=args.traffic_mode,
        duration=args.duration,
        fast_lane_safety_multiplier=args.fast_lane_safety_multiplier,
        fast_lane_min_ttc=args.fast_lane_min_ttc,
        fast_lane_accel_gap_multiplier=args.fast_lane_accel_gap_multiplier,
    )


def make_eval_env(
    seed: int,
    args: argparse.Namespace,
    render_mode: str,
    video_dir: Optional[Path] = None,
):
    actual_render_mode = "rgb_array" if video_dir is not None else render_mode
    if actual_render_mode == "human" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        actual_render_mode = "rgb_array"

    env = build_env(
        seed=seed,
        render_mode=actual_render_mode,
        eval_weather=args.eval_weather,
        train_mode=False,
        continuous_action=args.continuous_action,
        enable_action_noise=args.eval_action_noise,
        enable_overtake_assist=not args.no_overtake_assist,
        traffic_mode=args.traffic_mode,
        duration=args.duration,
        fast_lane_safety_multiplier=args.fast_lane_safety_multiplier,
        fast_lane_min_ttc=args.fast_lane_min_ttc,
        fast_lane_accel_gap_multiplier=args.fast_lane_accel_gap_multiplier,
    )

    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(env, video_folder=str(video_dir), episode_trigger=lambda episode_id: True)

    return env


def model_stem(args: argparse.Namespace) -> str:
    suffix = "continuous" if args.continuous_action else "discrete"
    return f"ppo_highway_longtail_realistic_v3_safety_{suffix}_{args.traffic_mode}"


def train(args: argparse.Namespace) -> Path:
    set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    log_dir = Path(args.log_dir)
    model_dir = Path(args.model_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    def env_fn():
        return make_train_env(args.seed, args)

    train_env = make_vec_env(
        env_fn,
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(log_dir / "monitor"),
    )

    eval_env = make_eval_env(
        args.seed + 10_000,
        args=args,
        render_mode="rgb_array",
        video_dir=None,
    )

    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    ent_coef = args.ent_coef
    if args.continuous_action and args.ent_coef == 0.0:
        ent_coef = 0.01
    elif not args.continuous_action and args.ent_coef == 0.0:
        # A little exploration helps the policy discover lane changes.
        ent_coef = 0.003

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir),
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=args.device,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(model_dir / "checkpoints"),
        name_prefix=model_stem(args),
    )

    callback = CallbackList([eval_callback, checkpoint_callback])

    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=args.progress_bar)

    final_model_path = model_dir / model_stem(args)
    model.save(str(final_model_path))

    train_env.close()
    eval_env.close()
    return final_model_path.with_suffix(".zip")


def evaluate(model_path: Path, args: argparse.Namespace) -> None:
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    video_dir = Path(args.video_dir) if args.video_dir else None

    if args.render_mode == "human" and not has_display and video_dir is None:
        video_dir = Path("videos")

    env = make_eval_env(
        seed=args.seed + 20_000,
        args=args,
        render_mode=args.render_mode,
        video_dir=video_dir,
    )

    model = PPO.load(str(model_path), device=args.device)
    live_render = args.render_mode == "human" and video_dir is None and has_display

    for episode in range(args.eval_episodes):
        obs, info = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        episode_reward = 0.0
        steps = 0

        raw_counter: Counter[str] = Counter()
        effective_counter: Counter[str] = Counter()
        assist_counter: Counter[str] = Counter()
        safety_counter: Counter[str] = Counter()
        block_counter: Counter[str] = Counter()
        lane_counter: Counter[str] = Counter()

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            steps += 1

            if args.debug_eval:
                raw_counter[str(info.get("raw_action", "n/a"))] += 1
                effective_counter[str(info.get("effective_action", "n/a"))] += 1
                assist_reason = str(info.get("assist_reason", ""))
                safety_reason = str(info.get("safety_shield_reason", ""))
                block_reason = str(info.get("lane_change_block_reason", ""))
                lane_counter[str(info.get("lane_id", "n/a"))] += 1
                if assist_reason:
                    assist_counter[assist_reason] += 1
                if safety_reason:
                    safety_counter[safety_reason] += 1
                if block_reason:
                    block_counter[block_reason] += 1

            if live_render:
                env.render()
                time.sleep(1.0 / 15.0)

        weather_value = info.get("weather_intensity", None)
        distance_value = info.get("safe_following_distance", None)
        weather_text = f"{float(weather_value):.2f}" if weather_value is not None else "n/a"
        distance_text = f"{float(distance_value):.1f}" if distance_value is not None else "n/a"

        print(
            f"eval_episode={episode + 1} "
            f"reward={episode_reward:.2f} "
            f"steps={steps} "
            f"weather={weather_text} "
            f"safe_distance={distance_text}"
        )

        if args.debug_eval:
            print(f"  raw_actions={dict(raw_counter)}")
            print(f"  effective_actions={dict(effective_counter)}")
            print(f"  assist={dict(assist_counter)}")
            print(f"  safety={dict(safety_counter)}")
            print(f"  blocked={dict(block_counter)}")
            print(f"  lanes={dict(lane_counter)}")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO on highway-v0 with active overtaking plus fast-lane safety-distance and acceleration limits."
    )

    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-dir", type=str, default="runs/ppo_highway_longtail_realistic_v3_safety")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Optional explicit .zip model path for evaluation or continued workflows.")

    parser.add_argument("--duration", type=int, default=80,
                        help="Episode duration in simulation seconds. This controls evaluation video length.")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array"])
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-weather", type=float, default=0.55, help="0=dry, 1=very rainy for evaluation")
    parser.add_argument("--traffic-mode", type=str, default="realistic", choices=["simple", "realistic", "dense"])
    parser.add_argument("--fast-lane-safety-multiplier", type=float, default=1.55,
                        help="Multiplier for following distance in the left-most overtaking lane.")
    parser.add_argument("--fast-lane-min-ttc", type=float, default=3.2,
                        help="Minimum time-to-collision in seconds before the fast-lane safety shield brakes.")
    parser.add_argument("--fast-lane-accel-gap-multiplier", type=float, default=1.35,
                        help="FASTER is blocked in the fast lane unless the front gap exceeds this multiplier times the fast-lane safe distance.")

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--continuous-action", action="store_true", help="Use continuous throttle/steering. Requires a new model.")

    parser.add_argument("--no-overtake-assist", action="store_true", help="Disable rule-based overtaking/return-right assist.")
    parser.add_argument("--no-action-noise", action="store_true", help="Disable small training-time actuation noise.")
    parser.add_argument("--eval-action-noise", action="store_true", help="Enable action noise during evaluation. Usually keep this off.")
    parser.add_argument("--debug-eval", action="store_true", help="Print raw/effective action counts during evaluation.")

    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--progress-bar", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = Path(args.model_dir) / f"{model_stem(args)}.zip"

    if not args.skip_train:
        model_path = train(args)
    elif not model_path.exists():
        raise FileNotFoundError(
            f"Missing model file: {model_path}. "
            "Train first, pass --model-path, or make sure --continuous-action and --traffic-mode match the model type."
        )

    if not args.skip_eval:
        evaluate(model_path, args)


if __name__ == "__main__":
    main()
