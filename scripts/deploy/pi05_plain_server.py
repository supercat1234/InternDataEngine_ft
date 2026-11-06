#!/usr/bin/env python3
"""Plain LeRobot 0.5.1 PI05 inference. No robot, RTC, AC or motion smoothing."""

import argparse
import base64
import io
import json
from pathlib import Path
import threading
import time
import urllib.request

import numpy as np


def validate_observation(observation, features):
    if set(observation) != set(features):
        raise ValueError(f"Expected observation keys: {sorted(features)}")
    result = {}
    for key, shape in features.items():
        value = np.asarray(observation[key])
        if key.startswith("observation.images."):
            channels, height, width = shape
            if value.shape != (height, width, channels) or value.dtype != np.uint8:
                raise ValueError(f"{key}: expected RGB uint8 HWC {(height, width, channels)}")
        else:
            value = value.astype(np.float32)
            if value.shape != tuple(shape) or not np.isfinite(value).all():
                raise ValueError(f"{key}: expected finite vector {shape}")
        result[key] = np.ascontiguousarray(value)
    return result


class PlainPI05:
    def __init__(self, checkpoint, tokenizer_path, device):
        import torch
        from transformers import AutoTokenizer
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            policy_action_to_transition, transition_to_policy_action,
        )

        class CheckedPI05(PI05Policy):
            def load_state_dict(self, *args, **kwargs):
                result = super().load_state_dict(*args, **kwargs)
                self._checkpoint_loaded = True
                return result

        root = Path(checkpoint).resolve()
        for name in ("config.json", "model.safetensors", "policy_preprocessor.json",
                     "policy_postprocessor.json"):
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)
        config = PreTrainedConfig.from_pretrained(str(root), local_files_only=True)
        if config.type != "pi05":
            raise ValueError("This example supports LeRobot PI05 checkpoints only")
        config.device = device
        config.rtc_config = None
        self.device = torch.device(device)
        self.policy = CheckedPI05.from_pretrained(
            str(root), config=config, local_files_only=True, strict=True,
        )
        # LeRobot 0.5.1 catches weight-loading exceptions internally. Refuse to
        # serve an initialized model if strict checkpoint loading did not finish.
        if not getattr(self.policy, "_checkpoint_loaded", False):
            raise RuntimeError("Checkpoint was not fully loaded; see loading errors above")
        self.policy.to(self.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.pre = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=str(root),
            config_filename="policy_preprocessor.json",
            overrides={
                "tokenizer_processor": {"tokenizer_name": tokenizer_path, "tokenizer": tokenizer},
                "device_processor": {"device": device},
            },
        )
        self.post = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=str(root),
            config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        self.features = {k: list(v.shape) for k, v in config.input_features.items()}
        self.action_dim = config.output_features["action"].shape[0]
        self.chunk_size = config.chunk_size
        self.lock = threading.Lock()

    def info(self):
        return {"model_type": "pi05", "rtc_enabled": False,
                "input_features": self.features, "action_dim": self.action_dim,
                "chunk_size": self.chunk_size}

    def infer(self, observation, task):
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        observation = validate_observation(observation, self.features)
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a nonempty string")
        with self.lock, torch.inference_mode():
            started = time.perf_counter()
            prepared = prepare_observation_for_inference(observation, self.device, task)
            # Use the fine-tuned checkpoint's normalization and unnormalization.
            chunk = self.policy.predict_action_chunk(self.pre(prepared))
            actions = self.post(chunk).detach().float().cpu().numpy()[0]
            if actions.shape != (self.chunk_size, self.action_dim) or not np.isfinite(actions).all():
                raise RuntimeError("Model returned invalid actions")
            return {"actions": actions.tolist(), "shape": list(actions.shape),
                    "inference_seconds": time.perf_counter() - started}


def create_app(engine):
    from flask import Flask, jsonify, request
    from werkzeug.exceptions import HTTPException

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

    @app.get("/health")
    def health():
        return jsonify(engine.info())

    @app.post("/infer")
    def infer():
        body = request.get_json()
        if not isinstance(body, dict):
            raise ValueError("Expected JSON object")
        try:
            raw = base64.b64decode(body["observation_npz"], validate=True)
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                observation = {key: data[key] for key in data.files}
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError("Invalid observation_npz") from exc
        return jsonify(engine.infer(observation, body.get("task")))

    @app.errorhandler(ValueError)
    def invalid_request(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(Exception)
    def server_error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        app.logger.exception("Inference failed")
        return jsonify(error="Inference failed; see server log"), 500

    return app


def infer_remote(url, observation, task, timeout=60):
    """Send raw RGB uint8 HWC images and unnormalized float32 state."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **observation)
    payload = {"task": task, "observation_npz": base64.b64encode(buffer.getvalue()).decode("ascii")}
    request = urllib.request.Request(
        url.rstrip("/") + "/infer", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    # Local inference should not be routed through a shell HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        result = json.load(response)
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2 or not np.isfinite(actions).all():
        raise ValueError("Invalid server actions")
    return actions, result["inference_seconds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--tokenizer", required=True)
    serve.add_argument("--device", default="cuda")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8097)
    infer = commands.add_parser("infer", help="Offline observation request; never controls hardware")
    infer.add_argument("--observation", required=True, help="NPZ with checkpoint observation keys")
    infer.add_argument("--task", required=True)
    infer.add_argument("--url", default="http://127.0.0.1:8097")
    infer.add_argument("--output", default="/tmp/pi05_actions.npy")
    args = parser.parse_args()
    if args.command == "serve":
        engine = PlainPI05(args.checkpoint, args.tokenizer, args.device)
        create_app(engine).run(host=args.host, port=args.port, debug=False, use_reloader=False)
    else:
        with np.load(args.observation, allow_pickle=False) as data:
            observation = {key: data[key] for key in data.files}
        actions, duration = infer_remote(args.url, observation, args.task)
        np.save(args.output, actions)
        print(f"actions={actions.shape}, inference={duration:.3f}s, saved={args.output}")


if __name__ == "__main__":
    main()
