import time
import uuid
import logging
from typing import List, Dict, Optional
import numpy as np
from collections import deque
import threading
import json
try:
    from HailoInferClass import HailoInfer
except Exception:
    HailoInfer = None

from config import Config
cfg = Config()
# Load list of labels
with open(cfg.class_labels_json, "r") as f:
    labels_list = json.load(f)

# Create dictionary: index ? label
class_names = {idx: label for idx, label in enumerate(labels_list)}




class HailoWrapper:
    def __init__(self, hef_path, batch_size, input_shape, logger, profiler=None, profile_interval: float = 5.0):
        self.hef_path = hef_path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.logger = logger
        self.profiler = profiler
        self.profile_interval = profile_interval

        if HailoInfer is None:
            #logger.warning("HailoInfer not importable; running in mock mode")
            self.model = None
        else:
            try:
                self.model = HailoInfer(
                    hef_path=hef_path,
                    batch_size=batch_size,
                    filenames={},
                    input_type="UINT8",
                    output_type="UINT8",
                )
            except Exception as e:
                logger.warning("Failed to create HailoInfer: %s", e)
                self.model = None

        self.mock_mode = self.model is None

        self.lock = threading.Lock()
        self.total_batches_submitted = 0
        self.total_samples_submitted = 0

        self.pending_batches: Dict[str, Dict] = {}
        self.hailo_device_latencies = deque(maxlen=1000)
        self.hailo_host_enqueue_latencies = deque(maxlen=1000)
        self.hailo_end_to_end_latencies = deque(maxlen=1000)
        self._last_profile_log = time.perf_counter()

    def close(self):
        if self.model:
            try:
                self.model.close()
            except Exception as e:
                self.logger.warning("Error closing Hailo: %s", e)
    def inference_callback(self, infer_results=None, bindings_list=None, batch_meta: Optional[Dict] = None, **kwargs):
        """
        Callback invoked by Hailo async execution or mock path.
        Records device/end-to-end latency and uses sample metadata to compute
        frame-to-result timings before running the lightweight prediction parsing.
        """
        device_end_time = time.perf_counter()
        batch_id = None
        sample_metas = []
        if batch_meta:
            batch_id = batch_meta.get("batch_id")
            sample_metas = batch_meta.get("samples") or []

        if bindings_list is None:
            self.logger.warning("No bindings_list received in callback.")
            return

        # Fallback: derive batch_id from bindings metadata if not provided
        if batch_id is None and bindings_list:
            meta0 = bindings_list[0].get("metadata") if isinstance(bindings_list[0], dict) else None
            if meta0:
                batch_id = meta0.get("batch_id") or meta0.get("sample_meta", {}).get("batch_id")

        pending = self.pending_batches.pop(batch_id, None)
        enqueue_time = None
        device_start_time = None
        if pending:
            enqueue_time = pending.get("enqueue_time")
            device_start_time = pending.get("device_start_time")
            if not sample_metas:
                sample_metas = pending.get("metas", [])
        if batch_meta and enqueue_time is None:
            enqueue_time = batch_meta.get("enqueue_time")
        if batch_meta and device_start_time is None:
            device_start_time = batch_meta.get("device_start_time")

        device_ms = None
        if device_start_time is not None:
            device_ms = (device_end_time - device_start_time) * 1000.0
            self.hailo_device_latencies.append(device_ms)
            if self.profiler:
                self.profiler.record("hailo_device_ms", device_ms)

        end_to_end_ms = None
        if enqueue_time is not None:
            end_to_end_ms = (device_end_time - enqueue_time) * 1000.0
            self.hailo_end_to_end_latencies.append(end_to_end_ms)
            if self.profiler:
                self.profiler.record("hailo_end_to_end_ms", end_to_end_ms)

        # Per-sample end-to-end stats (frame capture -> classification)
        if self.profiler and sample_metas:
            for sm in sample_metas:
                frame_perf = sm.get("frame_perf_counter")
                crop_done = sm.get("crop_done_time")
                if frame_perf is not None:
                    self.profiler.record("frame_to_result_ms", (device_end_time - frame_perf) * 1000.0)
                if frame_perf is not None and crop_done is not None:
                    self.profiler.record("frame_to_crop_ms", (crop_done - frame_perf) * 1000.0)
                if crop_done is not None and enqueue_time is not None:
                    self.profiler.record("crop_to_hailo_enqueue_ms", max(0.0, (enqueue_time - crop_done) * 1000.0))
                if enqueue_time is not None:
                    self.profiler.record("hailo_enqueue_to_result_ms", (device_end_time - enqueue_time) * 1000.0)

        # Keep a short periodic log of Hailo latencies
        if self.profiler:
            self.profiler.maybe_log(
                self.logger,
                keys=[
                    "hailo_host_enqueue_ms",
                    "hailo_device_ms",
                    "hailo_end_to_end_ms",
                    "frame_to_result_ms",
                ],
            )

        # Existing prediction parsing (lightweight, keeps behaviour intact)
        for info in bindings_list:
            binding = info.get("binding")

            try:
                if hasattr(binding, "output"):
                    output_buffers = binding.output().get_buffer()
                elif isinstance(binding, dict) and "output" in binding:
                    output_buffers = binding["output"]()
                else:
                    self.logger.error("Unsupported binding object in callback.")
                    continue
            except Exception as e:
                self.logger.error(f"Failed to get output buffer: {e}")
                continue

            # Handle dict or array output
            if isinstance(output_buffers, dict):
                out = list(output_buffers.values())[0]
            else:
                out = output_buffers

            # Convert to float32 flat vector
            logits = out.flatten().astype(np.float32)

            # Softmax
            exp_scores = np.exp(logits - np.max(logits))
            probs = exp_scores / np.sum(exp_scores)

            # Top-5
            top5_idx = probs.argsort()[-5:][::-1]
            top1_idx = int(top5_idx[0])
            top1_conf = float(probs[top1_idx] * 100.0)

            # Map class ID → label
            pred_label = class_names.get(top1_idx, f"class_{top1_idx}")

            # Log prediction (keep at debug to avoid noise)
            # self.logger.debug("[HAILO] Pred: %s (%.2f%%)", pred_label, top1_conf)


    def submit_batch(self, frames: List[np.ndarray], metas: List[Dict]):
        batch_id = str(uuid.uuid4())
        batch_meta = {
            "batch_id": batch_id,
            "samples": metas,
            "enqueue_time": time.perf_counter(),
        }

        t0 = time.perf_counter()
        self.pending_batches[batch_id] = {
            "enqueue_time": batch_meta["enqueue_time"],
            "device_start_time": t0,
            "metas": metas,
        }

        try:
            if self.model:
                try:
                    self.model.run_async(frames, self.inference_callback, batch_meta=batch_meta, sample_metas=metas)
                except TypeError:
                    self.model.run_async(frames, self.inference_callback)
            else:
                def _mock_job():
                    start_t = time.perf_counter()
                    time.sleep(0.02 + 0.005 * len(frames))
                    bindings_list = [
                        {
                            "binding": {"output": lambda: np.random.rand(1000)},
                            "metadata": {"batch_id": batch_id, "sample_index": i},
                        }
                        for i in range(len(frames))
                    ]
                    self.inference_callback(bindings_list=bindings_list, batch_meta={**batch_meta, "device_start_time": start_t})

                threading.Thread(target=_mock_job, daemon=True).start()

        except Exception as e:
            self.logger.error("Hailo run_async failed: %s", e)

        t1 = time.perf_counter()
        host_enqueue_ms = (t1 - t0) * 1000.0
        self.hailo_host_enqueue_latencies.append(host_enqueue_ms)
        if self.profiler:
            self.profiler.record("hailo_host_enqueue_ms", host_enqueue_ms)
            self.profiler.maybe_log(
                self.logger,
                keys=[
                    "hailo_host_enqueue_ms",
                    "hailo_device_ms",
                    "hailo_end_to_end_ms",
                ],
            )
        self.total_batches_submitted += 1
        self.total_samples_submitted += len(frames)

        return host_enqueue_ms
