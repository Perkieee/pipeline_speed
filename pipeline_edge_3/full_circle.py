import argparse
import logging
import signal
import sys
import time

import numpy as np

from config import Config
from edge_detector import EdgeDetector
from crop_batch_manager import CropBatchManager
from camera_manager import CameraManager
from profiler_utils import ProfilingManager

# Optional: Only import if you already have a Hailo inference runner
try:
    from hailo_infer import HailoWrapper
    USE_HAILO = True
except ImportError:
    HailoWrapper = None
    USE_HAILO = False


# ------------------------------------------------------------
# Graceful Kill Flag
# ------------------------------------------------------------
RUNNING = True


def handle_sigint(signum, frame):
    global RUNNING
    RUNNING = False
    print("\nShutting down...")


signal.signal(signal.SIGINT, handle_sigint)


def parse_args():
    parser = argparse.ArgumentParser(description="Full pipeline runner with profiling/test modes.")
    parser.add_argument("--profile-mode", choices=["full", "crop_only", "hailo_only"], help="Select profiling mode.")
    parser.add_argument("--profile-disable", action="store_true", help="Disable profiling instrumentation.")
    parser.add_argument("--profile-dump-path", type=str, help="Optional JSON dump path for profiling summary.")
    parser.add_argument("--run-time", type=float, help="Limit runtime in seconds.")
    parser.add_argument("--test-frames", type=int, help="Limit number of frames/crops for test modes.")
    parser.add_argument("--edge-mode", choices=["accuracy", "throughput"], help="Edge detector mode (accuracy saves crops).")
    parser.add_argument("--edge-save-dir", type=str, help="Directory to save edge detector crops (accuracy mode).")
    parser.add_argument("--show-mask-preview", action="store_true", help="Display the binary edge mask window.")
    return parser.parse_args()


def run_crop_only(cfg: Config, profiler: ProfilingManager, logger: logging.Logger) -> None:
    """
    Synthetic crop/batch benchmark without camera or Hailo/detector.
    """
    cbm = CropBatchManager(cfg, cfg.hailo_input_shape, logger, profiler=profiler)
    start = time.perf_counter()
    frames_sent = 0
    crops_processed = 0
    try:
        while True:
            if cfg.run_time and (time.perf_counter() - start) >= cfg.run_time:
                break
            if cfg.test_frames and frames_sent >= cfg.test_frames:
                break

            frame = np.random.randint(0, 255, (cfg.preview_size[1], cfg.preview_size[0], 3), dtype=np.uint8)
            detections = [
                (0.1, 0.1, 0.5, 0.5, 0.9),
                (0.4, 0.4, 0.9, 0.9, 0.8),
            ]
            meta = {
                "frame_perf_counter": time.perf_counter(),
                "frame_timestamp": time.time(),
            }
            cbm.submit_frame_detections(frames_sent, frame, detections, frame_meta=meta)
            frames_sent += 1

            while True:
                batch = cbm.get_next_batch(timeout=0.01)
                if batch is None:
                    break
                crops_processed += len(batch["frames"])

    finally:
        cbm.stop()
        profiler.dump_all_stats(logger, cfg.profile_dump_path)
        logger.info("Crop-only test complete: frames=%d crops=%d", frames_sent, crops_processed)


def run_hailo_only(cfg: Config, profiler: ProfilingManager, logger: logging.Logger) -> None:
    """
    Host + device latency benchmark using synthetic inputs only.
    """
    hailo = HailoWrapper(
        cfg.hef_path,
        cfg.hailo_batch_size,
        cfg.hailo_input_shape,
        logger,
        profiler=profiler,
        profile_interval=cfg.profile_log_interval_sec,
    )
    start = time.perf_counter()
    batches = 0
    samples = 0
    try:
        while True:
            if cfg.run_time and (time.perf_counter() - start) >= cfg.run_time:
                break
            if cfg.test_frames and samples >= cfg.test_frames:
                break

            frames = [
                np.random.randint(0, 255, cfg.hailo_input_shape, dtype=np.uint8)
                for _ in range(cfg.hailo_batch_size)
            ]
            now = time.perf_counter()
            metas = [
                {
                    "frame_perf_counter": now,
                    "crop_done_time": now,
                    "frame_idx": samples + i,
                    "det_idx": i,
                }
                for i in range(len(frames))
            ]
            hailo.submit_batch(frames, metas)
            batches += 1
            samples += len(frames)

    finally:
        hailo.close()
        profiler.dump_all_stats(logger, cfg.profile_dump_path)
        logger.info("Hailo-only test complete: batches=%d samples=%d", batches, samples)


def run_full_pipeline(cfg: Config, profiler: ProfilingManager, logger: logging.Logger) -> None:
    # Initialize modules
    detector = EdgeDetector(
        mode=cfg.edge_mode,
        crop_save_dir=cfg.edge_crop_save_dir,
        lower_yellow=cfg.edge_lower_yellow,
        upper_yellow=cfg.edge_upper_yellow,
        kernel_size=cfg.edge_kernel_size,
        min_area_ratio=cfg.edge_min_contour_area_ratio,
        min_area_floor=cfg.edge_min_contour_area_floor,
        margin_pixels=cfg.edge_margin_pixels,
        save_crops=cfg.edge_save_crops,
        show_mask=cfg.show_mask_preview,
        logger=logger.getChild("EdgeDetector"),
    )
    logger.info("Edge detector initialized (mode=%s, save_crops=%s)", cfg.edge_mode, detector.save_crops)

    logger.info("Initializing CropBatchManager...")
    cbm = CropBatchManager(cfg, cfg.hailo_input_shape, logger, profiler=profiler)

    hailo = None
    if USE_HAILO:
        hailo = HailoWrapper(
            cfg.hef_path,
            cfg.hailo_batch_size,
            cfg.hailo_input_shape,
            logger,
            profiler=profiler,
            profile_interval=cfg.profile_log_interval_sec,
        )

    logger.info("Initializing CameraManager...")
    cam = CameraManager(detector, cbm, cfg=cfg, profiler=profiler)

    # Start camera capture loop
    cam.start()
    images_processed = 0
    t_start = time.perf_counter()

    global RUNNING
    while RUNNING:
        if cfg.test_mode and images_processed >= cfg.test_frames:
            logger.info("Test frame limit reached; requesting stop.")
            RUNNING = False
            break
        if cfg.run_time and (time.perf_counter() - t_start) >= cfg.run_time:
            RUNNING = False

        batch = cbm.get_next_batch(timeout=0.1)
        if batch is None:
            continue
        images_processed += len(batch["frames"])

        if USE_HAILO and hailo:
            hailo.submit_batch(batch["frames"], batch["metas"])
        else:
            logger.info(
                "Batch %s | size=%d | first meta=%s",
                batch["batch_id"][:8],
                batch["size"],
                batch["metas"][0],
            )

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    fps = images_processed / elapsed if elapsed > 0 else 0

    logger.info("Total images processed: %d", images_processed)
    logger.info("Total time: %.2f sec", elapsed)
    logger.info("Pipeline throughput: %.2f images/second", fps)

    # Shutdown
    logger.info("Stopping components...")
    cam.stop()
    cbm.stop()

    if USE_HAILO and hailo:
        hailo.close()

    profiler.dump_all_stats(logger, cfg.profile_dump_path)
    logger.info("Shutdown complete.")
    sys.exit(0)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("MAIN")

    logger.info("Loading configuration...")
    cfg = Config()
    if args.profile_mode:
        cfg.profile_mode = args.profile_mode
    if args.profile_disable:
        cfg.profile_enabled = False
    if args.profile_dump_path:
        cfg.profile_dump_path = args.profile_dump_path
    if args.run_time is not None:
        cfg.run_time = args.run_time
    if args.test_frames is not None:
        cfg.test_frames = args.test_frames
    if args.edge_mode:
        cfg.edge_mode = args.edge_mode
    if args.edge_save_dir:
        cfg.edge_crop_save_dir = args.edge_save_dir
    if args.show_mask_preview:
        cfg.show_mask_preview = True

    profiler = ProfilingManager(
        enabled=cfg.profile_enabled,
        max_samples=cfg.profile_max_samples,
        log_interval=cfg.profile_log_interval_sec,
    )

    logger.info("Selected mode: %s", cfg.profile_mode)
    if cfg.profile_mode == "crop_only":
        run_crop_only(cfg, profiler, logger)
    elif cfg.profile_mode == "hailo_only":
        run_hailo_only(cfg, profiler, logger)
    else:
        run_full_pipeline(cfg, profiler, logger)


# ------------------------------------------------------------
# Boots the application
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
