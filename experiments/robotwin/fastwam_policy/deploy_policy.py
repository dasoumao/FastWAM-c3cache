import json
import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.inference_timing import InferenceTimer

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _validate_c3cache_support(model: torch.nn.Module) -> None:
    cache_keys = (
        "c3cache_enabled",
        "c3cache_start_step",
        "c3cache_end_step",
        "c3cache_refresh_interval",
    )
    parameters = inspect.signature(model.infer_action).parameters
    unsupported = [
        key
        for key in cache_keys
        if key not in parameters
        or parameters[key].kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    if unsupported or not callable(getattr(model, "reset_c3cache", None)):
        raise ValueError(
            f"c3cache_enabled=true requires {type(model).__name__}.infer_action "
            f"to explicitly accept {', '.join(cache_keys)} and provide reset_c3cache(). "
            f"Unsupported infer_action keyword parameters: {unsupported}. "
            "This model variant does not support C3ache."
        )


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        *,
        c3cache_enabled: bool = False,
        c3cache_start_step: int = 0,
        c3cache_end_step: int = 6,
        c3cache_refresh_interval: int = 4,
        compile_action_infer: bool = False,
        inference_diagnostics_enabled: bool = False,
        inference_diagnostics_every_n_chunks: int = 1,
        inference_diagnostics_max_chunks: int = 200,
        inference_diagnostics_cuda_events: bool = True,
        timing_output_dir: Optional[Path] = None,
        task_name: Optional[str] = None,
        task_config: Optional[str] = None,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()
        if compile_action_infer:
            compile_parameter = inspect.signature(self.model.infer_action).parameters.get(
                "compile_action_infer"
            )
            if compile_parameter is None or compile_parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise ValueError(
                    "compile_action_infer=true requires infer_action to accept "
                    "the compile_action_infer keyword."
                )
        if c3cache_enabled:
            _validate_c3cache_support(self.model)
            self.model.reset_c3cache()
        if inference_diagnostics_enabled:
            configure_diagnostics = getattr(self.model, "configure_inference_diagnostics", None)
            if not callable(configure_diagnostics) or not callable(
                getattr(self.model, "get_inference_diagnostics", None)
            ):
                raise ValueError(
                    "inference_diagnostics_enabled=true requires a model with "
                    "configure_inference_diagnostics() and get_inference_diagnostics()."
                )
            configure_diagnostics(
                enabled=True,
                every_n_chunks=int(inference_diagnostics_every_n_chunks),
                max_chunks=int(inference_diagnostics_max_chunks),
                cuda_events=bool(inference_diagnostics_cuda_events),
            )

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.c3cache_enabled = bool(c3cache_enabled)
        self.c3cache_start_step = int(c3cache_start_step)
        self.c3cache_end_step = int(c3cache_end_step)
        self.c3cache_refresh_interval = int(c3cache_refresh_interval)
        self.compile_action_infer = bool(compile_action_infer)
        self.timing_enabled = bool(timing_enabled)
        self.inference_diagnostics_enabled = bool(inference_diagnostics_enabled)
        self._diagnostics_max_chunks = int(inference_diagnostics_max_chunks)
        self.task_name = task_name
        self.task_config = task_config
        timing_tag = "".join(
            char if char.isalnum() or char in "_-" else "_"
            for char in str(task_config or "unspecified")
        )
        self._timing_output_path = (
            Path(timing_output_dir) / f"fastwam_inference_timing_{timing_tag}.jsonl"
            if timing_output_dir is not None else None
        )
        self._diagnostics_output_path = (
            Path(timing_output_dir) / f"fastwam_inference_diagnostics_{timing_tag}.jsonl"
            if timing_output_dir is not None else None
        )
        self._diagnostic_records_written = 0
        self._num_video_frames = int(num_video_frames)

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {
            "infer_s": 0.0,
            "infer_cuda_s": 0.0,
            "sim_s": 0.0,
            "infer_chunks": 0,
            "infer_cuda_chunks": 0,
        }

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        if self.compile_action_infer:
            infer_kwargs["compile_action_infer"] = True
        if self.c3cache_enabled:
            infer_kwargs.update(
                c3cache_enabled=True,
                c3cache_start_step=self.c3cache_start_step,
                c3cache_end_step=self.c3cache_end_step,
                c3cache_refresh_interval=self.c3cache_refresh_interval,
            )
        with torch.no_grad():
            with InferenceTimer(device=self.model.device, enabled=self.timing_enabled) as timer:
                pred = self.model.infer_action(**infer_kwargs)
        if (
            self.inference_diagnostics_enabled
            and self._diagnostic_records_written < self._diagnostics_max_chunks
        ):
            diagnostic_snapshot = self.model.get_inference_diagnostics()
            new_records = diagnostic_snapshot["records"][self._diagnostic_records_written:]
            if new_records:
                first_record = self._diagnostic_records_written == 0
                self._diagnostic_records_written += len(new_records)
                output_records = [dict(record) for record in new_records]
                if first_record:
                    output_records[0]["_task_metadata"] = diagnostic_snapshot["metadata"]
                if self._diagnostics_output_path is None:
                    for record in output_records:
                        print(f"FastWAM inference diagnostics: {json.dumps(record)}", flush=True)
                else:
                    self._diagnostics_output_path.parent.mkdir(parents=True, exist_ok=True)
                    with self._diagnostics_output_path.open("a", encoding="utf-8") as diagnostics_file:
                        for record in output_records:
                            diagnostics_file.write(json.dumps(record) + "\n")
        if self.timing_enabled:
            infer_s = timer.wall_seconds
            infer_cuda_s = timer.cuda_seconds
            assert infer_s is not None
            self._timing_rollout["infer_s"] += infer_s
            self._timing_rollout["infer_chunks"] += 1
            if infer_cuda_s is not None:
                self._timing_rollout["infer_cuda_s"] += infer_cuda_s
                self._timing_rollout["infer_cuda_chunks"] += 1
            timing_record = {
                "task_name": self.task_name,
                "task_config": self.task_config,
                "episode": self.episode_count,
                "chunk": self._timing_rollout["infer_chunks"],
                "infer_s": infer_s,
                "infer_cuda_s": infer_cuda_s,
                "cumulative_infer_s": self._timing_rollout["infer_s"],
                "cumulative_infer_cuda_s": (
                    self._timing_rollout["infer_cuda_s"] if infer_cuda_s is not None else None
                ),
                "cumulative_infer_cuda_chunks": self._timing_rollout["infer_cuda_chunks"],
                "c3cache": {
                    "enabled": self.c3cache_enabled,
                    "start_step": self.c3cache_start_step,
                    "end_step": self.c3cache_end_step,
                    "refresh_interval": self.c3cache_refresh_interval,
                },
            }
            if self.c3cache_enabled:
                stats_fn = getattr(self.model, "get_c3cache_stats", None)
                if callable(stats_fn):
                    timing_record["c3cache_stats"] = stats_fn()
            if self._timing_output_path is None:
                print(f"FastWAM inference timing: {json.dumps(timing_record)}", flush=True)
            else:
                self._timing_output_path.parent.mkdir(parents=True, exist_ok=True)
                with self._timing_output_path.open("a", encoding="utf-8") as timing_file:
                    timing_file.write(json.dumps(timing_record) + "\n")

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["infer_cuda_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0
        self._timing_rollout["infer_chunks"] = 0
        self._timing_rollout["infer_cuda_chunks"] = 0

    def get_timing_rollout(self) -> Dict[str, float | int | None]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "infer_cuda_s": (
                float(self._timing_rollout["infer_cuda_s"])
                if self._timing_rollout["infer_cuda_chunks"] else None
            ),
            "sim_s": float(self._timing_rollout["sim_s"]),
            "infer_chunks": int(self._timing_rollout["infer_chunks"]),
            "infer_cuda_chunks": int(self._timing_rollout["infer_cuda_chunks"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        if self.c3cache_enabled:
            self.model.reset_c3cache()
        self.episode_count += 1
        if self.inference_diagnostics_enabled:
            context_fn = getattr(self.model, "set_inference_diagnostic_context", None)
            if callable(context_fn):
                context_fn(
                    task_name=self.task_name,
                    task_config=self.task_config,
                    episode_index=self.episode_count,
                )
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    c3cache_enabled = _parse_bool(
        usr_args.get("c3cache_enabled", cfg.EVALUATION.get("c3cache_enabled", False))
    )
    compile_action_infer = _parse_bool(
        usr_args.get("compile_action_infer", cfg.EVALUATION.get("compile_action_infer", False))
    )
    c3cache_start_step = int(
        usr_args.get("c3cache_start_step", cfg.EVALUATION.get("c3cache_start_step", 0))
    )
    c3cache_end_step = int(
        usr_args.get("c3cache_end_step", cfg.EVALUATION.get("c3cache_end_step", 6))
    )
    c3cache_refresh_interval = int(
        usr_args.get("c3cache_refresh_interval", cfg.EVALUATION.get("c3cache_refresh_interval", 4))
    )
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", True))
    )
    inference_diagnostics_enabled = _parse_bool(
        usr_args.get(
            "inference_diagnostics_enabled",
            cfg.EVALUATION.get("inference_diagnostics_enabled", False),
        )
    )
    inference_diagnostics_every_n_chunks = int(
        usr_args.get(
            "inference_diagnostics_every_n_chunks",
            cfg.EVALUATION.get("inference_diagnostics_every_n_chunks", 1),
        )
    )
    inference_diagnostics_max_chunks = int(
        usr_args.get(
            "inference_diagnostics_max_chunks",
            cfg.EVALUATION.get("inference_diagnostics_max_chunks", 200),
        )
    )
    inference_diagnostics_cuda_events = _parse_bool(
        usr_args.get(
            "inference_diagnostics_cuda_events",
            cfg.EVALUATION.get("inference_diagnostics_cuda_events", True),
        )
    )

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        c3cache_enabled=c3cache_enabled,
        c3cache_start_step=c3cache_start_step,
        c3cache_end_step=c3cache_end_step,
        c3cache_refresh_interval=c3cache_refresh_interval,
        compile_action_infer=compile_action_infer,
        timing_enabled=timing_enabled,
        inference_diagnostics_enabled=inference_diagnostics_enabled,
        inference_diagnostics_every_n_chunks=inference_diagnostics_every_n_chunks,
        inference_diagnostics_max_chunks=inference_diagnostics_max_chunks,
        inference_diagnostics_cuda_events=inference_diagnostics_cuda_events,
        timing_output_dir=(
            Path(str(usr_args["eval_output_dir"]))
            if not _is_none_like(usr_args.get("eval_output_dir")) else None
        ),
        task_name=(
            str(usr_args["task_name"]) if not _is_none_like(usr_args.get("task_name")) else None
        ),
        task_config=(
            str(usr_args["task_config"])
            if not _is_none_like(usr_args.get("task_config")) else None
        ),
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
