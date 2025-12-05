import time
import uuid
import cv2
import queue
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque
import threading

class CropBatchManager:
    def __init__(self, cfg, hailo_input_shape, logger, profiler=None):
        self.cfg = cfg
        self.h_input_shape = hailo_input_shape
        self.logger = logger
        self.profiler = profiler

        self.crop_pool = ThreadPoolExecutor(max_workers=cfg.crop_thread_workers)

        # decoupled capacity to avoid tying queue depth to batch size
        self.crop_result_queue = queue.Queue(maxsize=cfg.crop_queue_capacity)

        self.batch_queue = queue.Queue(maxsize=cfg.queue_maxsize)

        self.crop_timings = deque(maxlen=1000)
        self.batch_prepare_timings = deque(maxlen=1000)
        self.crop_end_to_end_timings = deque(maxlen=1000)
        self.crops_dropped = 0
        # Track per-frame expected and remaining crops for per-frame batching/flush
        self.frame_expected_crops: Dict[int, int] = {}
        self.frame_remaining_crops: Dict[int, int] = {}
        self._frame_lock = threading.Lock()

        self._stop_event = False
        self._batch_thread = threading.Thread(
            target=self._batch_assembler_thread,
            daemon=True
        )
        self._batch_thread.start()
        self._last_profile_log = time.perf_counter()
        self.profile_interval = cfg.profile_log_interval_sec

    def stop(self):
        self._stop_event = True
        self.crop_pool.shutdown(wait=False)

    @staticmethod
    def crop_to_bbox(image, bbox_norm):
        H, W = image.shape[:2]
        xmin_n, ymin_n, xmax_n, ymax_n = bbox_norm

        x1 = int(max(0, xmin_n * W))
        y1 = int(max(0, ymin_n * H))
        x2 = int(min(W, xmax_n * W))
        y2 = int(min(H, ymax_n * H))

        if x1 >= x2 or y1 >= y2:
            return np.zeros((1, 1, 3), dtype=image.dtype)

        return np.ascontiguousarray(image[y1:y2, x1:x2]) #.copy()

    @staticmethod
    def resize_for_hailo(img, target_shape):
        h, w = target_shape[:2]
        out = cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img

        if out.dtype != np.uint8:
            out = out.astype(np.uint8)
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

        return out

    def submit_frame_detections(self, frame_idx, frame_rgb, detections, frame_meta: Optional[Dict] = None):
        if not detections:
            return

        # Record how many crops we expect for this frame to enable per-frame flushing
        with self._frame_lock:
            self.frame_expected_crops[frame_idx] = len(detections)
            self.frame_remaining_crops[frame_idx] = len(detections)

        for det_idx, det in enumerate(detections):
            bbox = det[:4]
            submit_t = time.perf_counter()

            fut: Future = self.crop_pool.submit(
                self._crop_and_resize_task,
                frame_rgb,
                bbox,
                submit_t,
            )
            fut.add_done_callback(
                lambda f, fi=frame_idx, di=det_idx, st=submit_t, fm=frame_meta: self._on_crop_done(f, fi, di, st, fm)
            )

    def _crop_and_resize_task(self, frame_rgb, bbox_norm, submit_time):
        t0 = time.perf_counter()
        roi = self.crop_to_bbox(frame_rgb, bbox_norm)
        roi_resized = self.resize_for_hailo(roi, self.h_input_shape)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        return roi_resized, bbox_norm, elapsed_ms

    def _on_crop_done(self, fut, frame_idx, det_idx, submit_time, frame_meta: Optional[Dict]):
        try:
            roi_resized, bbox_norm, crop_ms = fut.result()
        except Exception as e:
            self.logger.error("Crop task failed: %s", e)
            return

        done_time = time.perf_counter()
        total_ms = (done_time - submit_time) * 1000.0
        remaining = None
        expected = None
        # Update remaining count for this frame to know when the frame is complete
        with self._frame_lock:
            expected = self.frame_expected_crops.get(frame_idx)
            if frame_idx in self.frame_remaining_crops:
                new_remaining = max(0, self.frame_remaining_crops[frame_idx] - 1)
                self.frame_remaining_crops[frame_idx] = new_remaining
                remaining = new_remaining
        meta = {
            "frame_idx": frame_idx,
            "det_idx": det_idx,
            "bbox_norm": bbox_norm,
            "crop_enqueue_time": submit_time,
            "crop_done_time": done_time,
            "frame_expected": expected,
            "frame_remaining": remaining,
        }
        if frame_meta:
            meta["frame_perf_counter"] = frame_meta.get("frame_perf_counter")
            meta["frame_timestamp"] = frame_meta.get("frame_timestamp")
            meta["imx_inf_ms"] = frame_meta.get("imx_inf_ms")
            meta["imx_post_ms"] = frame_meta.get("imx_post_ms")
            meta["raw_metadata"] = frame_meta.get("raw_metadata")

        item = (frame_idx, det_idx, roi_resized, bbox_norm, done_time, crop_ms, meta)

        try:
            self.crop_result_queue.put(item, timeout=self.cfg.crop_put_timeout)
            self.crop_timings.append(crop_ms)
            self.crop_end_to_end_timings.append(total_ms)
            if self.profiler:
                self.profiler.record("crop_task_ms", crop_ms)
                self.profiler.record("crop_end_to_end_ms", total_ms)
        except queue.Full:
            self.crops_dropped += 1
            self.logger.warning("Crop queue full - dropping crop")

    def _batch_assembler_thread(self):
        partial = []
        last_time = time.perf_counter()
        first_enqueue_time = None
        current_frame_idx = None
        # We also track how many crops we believe are still outstanding for the current frame
        current_frame_remaining = None

        while not self._stop_event:
            try:
                item = self.crop_result_queue.get(timeout=0.1)
            except queue.Empty:
                # Timeout flush to avoid stalling if producer stops
                if partial and (time.perf_counter() - last_time) > 0.05:
                    self._flush(partial, first_enqueue_time)
                    partial = []
                    first_enqueue_time = None
                    current_frame_idx = None
                    current_frame_remaining = None
                    last_time = time.perf_counter()
                continue

            frame_idx, det_idx, roi_resized, bbox_norm, enqueue_time, crop_ms, meta = item

            # Enforce one batch per frame: if frame changes, flush existing partial first
            if current_frame_idx is not None and frame_idx != current_frame_idx:
                self._flush(partial, first_enqueue_time)
                partial = []
                first_enqueue_time = None
                current_frame_remaining = None

            if first_enqueue_time is None:
                first_enqueue_time = meta.get("crop_enqueue_time", enqueue_time)
            current_frame_idx = frame_idx
            current_frame_remaining = meta.get("frame_remaining")

            partial.append({"image": roi_resized, "meta": meta})

            # If we've collected all crops for this frame, flush immediately (primary per-frame trigger)
            if current_frame_remaining is not None and current_frame_remaining <= 0:
                self._flush(partial, first_enqueue_time)
                partial = []
                first_enqueue_time = None
                current_frame_idx = None
                current_frame_remaining = None
                last_time = time.perf_counter()
                continue

            # Hard cap for safety
            if len(partial) >= self.cfg.batch_max_size:
                self._flush(partial, first_enqueue_time)
                partial = []
                first_enqueue_time = None
                current_frame_idx = None
                last_time = time.perf_counter()
                current_frame_remaining = None
                continue

            # Time-based safety flush to bound queue_delay_ms; ensures first crop in batch
            # never waits longer than cfg.max_batch_delay_ms even if crops arrive slowly.
            if first_enqueue_time is not None:
                age_ms = (time.perf_counter() - first_enqueue_time) * 1000.0
                if age_ms >= self.cfg.max_batch_delay_ms:
                    self._flush(partial, first_enqueue_time)
                    partial = []
                    first_enqueue_time = None
                    current_frame_idx = None
                    current_frame_remaining = None
                    last_time = time.perf_counter()

        if partial:
            self._flush(partial, first_enqueue_time)

    def _flush(self, items, first_enqueue_time: Optional[float]):
        t0 = time.perf_counter()
        batch_id = str(uuid.uuid4())

        frames = [x["image"] for x in items]
        metas = [x["meta"] for x in items]
        assembled_time = time.perf_counter()
        queue_delay_ms = 0.0
        if first_enqueue_time is not None:
            queue_delay_ms = (assembled_time - first_enqueue_time) * 1000.0

        batch = {
            "batch_id": batch_id,
            "frames": frames,
            "metas": metas,
            "size": len(frames),
            "assembled_time": assembled_time,
            "first_enqueue_time": first_enqueue_time,
            "queue_delay_ms": queue_delay_ms,
        }
        for m in metas:
            m["batch_id"] = batch_id
            m["batch_assembled_time"] = assembled_time
            m["batch_queue_delay_ms"] = queue_delay_ms

        # Clean up per-frame counters once we've flushed that frame
        with self._frame_lock:
            if metas:
                frame_id = metas[0].get("frame_idx")
                if frame_id in self.frame_expected_crops:
                    self.frame_expected_crops.pop(frame_id, None)
                if frame_id in self.frame_remaining_crops:
                    self.frame_remaining_crops.pop(frame_id, None)

        try:
            self.batch_queue.put_nowait(batch)
        except queue.Full:
            self.logger.warning("Batch queue full - dropping batch")

        t1 = time.perf_counter()
        prep_ms = (t1 - t0) * 1000
        self.batch_prepare_timings.append(prep_ms)
        if self.profiler:
            self.profiler.record("batch_prepare_ms", prep_ms)
            self.profiler.record("batch_queue_delay_ms", queue_delay_ms)
            self.profiler.maybe_log(
                self.logger,
                stages=[
                    "crop_task_ms",
                    "crop_end_to_end_ms",
                    "batch_prepare_ms",
                    "batch_queue_delay_ms",
                ],
            )

    def get_next_batch(self, timeout=0.1):
        try:
            return self.batch_queue.get(timeout=timeout)
        except queue.Empty:
            return None
