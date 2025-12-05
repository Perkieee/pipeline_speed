# camera_manager.py
import threading
import time
import logging
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    from picamera2.controls import Controls
except Exception:
    Picamera2 = None  # fallback to None if library not available

from crop_batch_manager import CropBatchManager
# Note: detection backend and CropBatchManager are not imported here to keep this
# module testable and decoupled from specific models.

class CameraManager:
    """
    CameraManager
    -------------
    Responsibilities:
    - Initialize Picamera2 preview stream and start capture.
    - In a background thread: capture requests (frame + metadata), call the detector to
      extract detections, hand the frame + detections to CropBatchManager,
      and store frame-level metadata (timestamp + raw metadata) for later lookup.
    - Provide start()/stop() lifecycle and a small API to query frame metadata via frame index.
    """

    def __init__(
        self,
        detector,                       # instance implementing detect(...) or get_detections_from_metadata(...)
        crop_batch_manager,             # instance of CropBatchManager (must implement submit_frame_detections)
        cfg=None,
        profiler=None,
        preview_size: Optional[Tuple[int, int]] = (640, 360),
        show_preview: Optional[bool] = None,
        warmup_seconds: float = 2.0,
        run_time: Optional[float] = None,   # if set, camera will stop after run_time seconds
        logger: Optional[logging.Logger] = None,
    ):
        self.detector = detector
        self.crop_mgr = crop_batch_manager
        self.cfg = cfg
        self.profiler = profiler
        self.preview_size = preview_size if preview_size is not None else (cfg.preview_size if cfg else (640, 360))
        chosen_preview_flag = show_preview if show_preview is not None else (cfg.show_preview if cfg else True)
        self.show_preview = bool(chosen_preview_flag)
        self.show_mask_preview = bool(getattr(cfg, "show_mask_preview", False)) if cfg is not None else False
        self.warmup_seconds = warmup_seconds
        self.run_time = run_time if run_time is not None else (cfg.run_time if cfg else None)
        self.logger = logger or logging.getLogger("CameraManager")
        self.profile_interval = cfg.profile_log_interval_sec if cfg else 5.0

        # Camera and thread control
        self.picam2 = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Frame index counter (monotonic)
        self._frame_index = 0
        # Map frame_idx -> frame-level metadata (contains at least 'frame_timestamp' per your request)
        self._frame_metadata: Dict[int, Dict[str, Any]] = {}
        # Lock protecting access to frame metadata
        self._meta_lock = threading.Lock()
        self._last_profile_log = time.perf_counter()
        self._frames_since_log = 0
        # Camera run stats (per capture loop)
        self._cam_total_frames = 0
        self._cam_run_start_time: Optional[float] = None
        self._cam_run_elapsed = 0.0

    # ---------------- lifecycle ----------------
    def start(self) -> None:
        """
        Start Picamera2 and the capture thread.
        Raises RuntimeError if Picamera2 is not available.
        """
        if Picamera2 is None:
            raise RuntimeError("Picamera2 not available in this environment.")

        if self.picam2 is not None:
            self.logger.warning("Camera already started")
            return

        # Initialize camera
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": self.preview_size, "format": "BGR888"})
        self.picam2.configure(config)

        try:
            self.picam2.start()
        except Exception as e:
            # Clean up if start fails
            self.picam2 = None
            raise RuntimeError(f"Failed to start Picamera2: {e}") from e

        # Warmup AWB/AE if requested
        if self.warmup_seconds and self.warmup_seconds > 0:
            self.logger.info("Camera started: warming up AWB/AE for %.1f seconds...", float(self.warmup_seconds))
            time.sleep(float(self.warmup_seconds))

        # Reset control flags and start worker thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="CameraCaptureThread", daemon=True)
        self._thread.start()
        self.logger.info("Camera capture thread started.")

    def stop(self, join_timeout: float = 2.0) -> None:
        """
        Signal stop and wait for thread + camera to shut down.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                self.logger.warning("Camera thread did not exit within timeout.")
            self._thread = None

        # stop camera if running
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception as e:
                self.logger.exception("Error stopping Picamera2: %s", e)
            finally:
                self.picam2 = None

        self.logger.info("CameraManager stopped.")

    # ---------------- metadata API ----------------
    def get_frame_metadata(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        """
        Return a copy of the stored frame metadata (which includes 'frame_timestamp') for the given frame index.
        Returns None if not found.
        """
        with self._meta_lock:
            md = self._frame_metadata.get(frame_idx)
            return dict(md) if md is not None else None

    def get_camera_run_stats(self) -> Tuple[int, float]:
        """
        Return (total_frames, elapsed_seconds) for the most recent camera run.
        """
        return self._cam_total_frames, self._cam_run_elapsed

    # ---------------- internal capture loop ----------------
    def _capture_loop(self) -> None:
        """
        Main capture loop running in background thread.

        Flow per loop:
         - capture request (req)
         - obtain frame array (req.make_array('main') or fallback capture_array)
         - extract metadata via req.get_metadata()
         - store frame-level metadata (frame_timestamp if present)
         - call detector (edge/color or IMX) to produce detections
         - hand detections + frame to crop_batch_manager.submit_frame_detections(frame_idx, frame_rgb, detections)
         - optionally draw preview and show using OpenCV
         - release the request (req.release())
        """
        self._cam_run_start_time = time.perf_counter()
        self._cam_total_frames = 0
        start_time = self._cam_run_start_time

        while not self._stop_event.is_set():
            # Optionally exit after run_time (if provided)
            if self.run_time is not None and (time.perf_counter() - start_time) >= float(self.run_time):
                self.logger.info("CameraManager run_time reached; exiting capture loop.")
                break

            req = None
            try:
                t0_frame = time.perf_counter()
                # Capture a request (blocks until available from camera pipeline)
                req = self.picam2.capture_request()
                try:
                    frame_rgb = req.make_array("main")
                except Exception:
                    # Fallback API
                    frame_rgb = self.picam2.capture_array()

                # Obtain metadata from the same request so boxes match frame exactly
                raw_metadata = req.get_metadata() or {}
                t1_frame = time.perf_counter()
                # Determine a reasonable frame timestamp: prefer 'SensorTimestamp', fallback to perf_counter/time
                frame_timestamp = raw_metadata.get("SensorTimestamp") or raw_metadata.get("Timestamp") or time.time()

                # Record frame metadata for downstream lookups
                frame_idx = self._frame_index
                with self._meta_lock:
                    self._frame_metadata[frame_idx] = {
                        "frame_timestamp": frame_timestamp,
                        "raw_metadata": raw_metadata,
                        # High resolution reference for end-to-end latency (perf_counter at capture)
                        "frame_perf_counter": t1_frame,
                    }

                # --------------- Detection (edge/color or IMX backend) ---------------
                try:
                    detections, det_inf_ms, det_post_ms, debug = self._run_detection(frame_rgb, raw_metadata)
                except Exception as e:
                    self.logger.exception("Detector raised exception while getting detections: %s", e)
                    detections, det_inf_ms, det_post_ms, debug = [], 0.0, 0.0, {}

                # --------------- Submit detections to CropBatchManager ---------------
                try:
                    # crop_mgr will handle cropping/resizing asynchronously
                    frame_meta = self.get_frame_metadata(frame_idx) or {}
                    frame_meta.update({
                        "detector_inf_ms": det_inf_ms,
                        "detector_post_ms": det_post_ms,
                        # backward-compatible keys for any downstream consumers
                        "imx_inf_ms": det_inf_ms,
                        "imx_post_ms": det_post_ms,
                    })
                    self.crop_mgr.submit_frame_detections(frame_idx, frame_rgb, detections, frame_meta=frame_meta)
                except Exception as e:
                    self.logger.exception("CropBatchManager.submit_frame_detections failed: %s", e)

                t3_after_submit = time.perf_counter()
                capture_ms = (t1_frame - t0_frame) * 1000.0
                rest_ms = max(0.0, ((t3_after_submit - t1_frame) * 1000.0) - (det_inf_ms + det_post_ms))
                if self.profiler:
                    self.profiler.record("camera_capture_ms", capture_ms)
                    self.profiler.record("camera_to_crop_submit_ms", (t3_after_submit - t1_frame) * 1000.0)
                    self.profiler.record("detector_inference_ms", det_inf_ms)
                    self.profiler.record("detector_postprocess_ms", det_post_ms)
                    # legacy metric names to keep graphs/scripts working
                    self.profiler.record("imx_inference_ms", det_inf_ms)
                    self.profiler.record("imx_postprocess_ms", det_post_ms)
                    self.profiler.record("camera_other_ms", rest_ms)

                # --------------- Optional preview display ---------------
                if self.show_preview:
                    try:
                        # Draw boxes for quick visual debugging (non-blocking, lightweight)
                        disp = frame_rgb.copy()
                        H, W = disp.shape[:2]
                        for (xmin, ymin, xmax, ymax, score) in detections:
                            x1 = int(xmin * W); y1 = int(ymin * H)
                            x2 = int(xmax * W); y2 = int(ymax * H)
                            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{score:.2f}"
                            cv2.putText(disp, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        cv2.imshow("Camera Preview", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
                        if self.show_mask_preview and debug.get("mask") is not None:
                            cv2.imshow("Mask", debug["mask"])
                        if cv2.waitKey(1) & 0xFF == 27:
                            self.logger.info("Preview ESC pressed - requesting stop.")
                            self._stop_event.set()
                    except Exception:
                        # Ensure preview errors don't kill capture loop
                        self.logger.exception("Preview draw failed.")
                # release request buffers quickly so libcamera can reuse them
                try:
                    if req:
                        req.release()
                except Exception:
                    # ignore release errors
                    pass

                # Advance frame index (monotonic)
                self._frame_index += 1
                self._frames_since_log += 1
                self._cam_total_frames += 1

                # Periodic FPS + profiling summaries at configured interval
                now = time.perf_counter()
                if (now - self._last_profile_log) >= self.profile_interval:
                    elapsed = now - self._last_profile_log
                    fps = self._frames_since_log / elapsed if elapsed > 0 else 0.0
                    self.logger.info("Camera FPS over last %.1fs: %.2f", elapsed, fps)
                    if self.profiler:
                        self.profiler.maybe_log(
                            self.logger,
                            keys=[
                                "camera_capture_ms",
                                "detector_inference_ms",
                                "detector_postprocess_ms",
                                "camera_to_crop_submit_ms",
                            ],
                        )
                    self._last_profile_log = now
                    self._frames_since_log = 0

            except Exception as e:
                # Broad exception to avoid capture thread death; log and continue or break if fatal
                self.logger.exception("Capture loop exception: %s", e)
                # Attempt to release request if allocated
                try:
                    if req:
                        req.release()
                except Exception:
                    pass
                # Brief sleep to avoid tight error loop
                time.sleep(0.01)
                continue

        # capture loop finished; clean up preview windows if used
        if self.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if self._cam_run_start_time is not None:
            self._cam_run_elapsed = time.perf_counter() - self._cam_run_start_time
        else:
            self._cam_run_elapsed = 0.0

        if self._cam_run_elapsed > 0 and self._cam_total_frames > 0:
            cam_fps = self._cam_total_frames / self._cam_run_elapsed
            self.logger.info(
                "Camera frame summary: total frames: %d  total time: %.2f sec  camera FPS: %.2f",
                self._cam_total_frames,
                self._cam_run_elapsed,
                cam_fps,
            )
            if self.profiler:
                self.profiler.record("camera_run_seconds", self._cam_run_elapsed * 1000.0)
                self.profiler.record("camera_fps", cam_fps)

        self.logger.info("Camera capture thread exiting.")

    def _run_detection(self, frame_rgb: np.ndarray, raw_metadata: Dict[str, Any]):
        """
        Support both the legacy IMX500 API (get_detections_from_metadata) and a modern
        detect(frame_rgb, metadata) API used by the edge detector.
        """
        if hasattr(self.detector, "detect"):
            result = self.detector.detect(frame_rgb, raw_metadata)
            if isinstance(result, tuple):
                if len(result) == 4:
                    return result
                if len(result) == 3:
                    detections, det_ms, post_ms = result
                    return detections, det_ms, post_ms, {}
            # Fallback: detector returned detections only
            return result, 0.0, 0.0, {}

        if hasattr(self.detector, "get_detections_from_metadata"):
            detections, det_ms, post_ms = self.detector.get_detections_from_metadata(raw_metadata)
            return detections, det_ms, post_ms, {}

        raise RuntimeError("Detector does not implement detect(...) or get_detections_from_metadata(...).")
