from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class Config:
    # Edge detector configuration
    edge_mode: str = "accuracy"  # "accuracy" saves crops, "throughput" skips saving
    edge_crop_save_dir: str = "/home/sir/Kevin_Walter/tests/edge_real_time/crops"
    edge_lower_yellow: Tuple[int, int, int] = (18, 30, 120)
    edge_upper_yellow: Tuple[int, int, int] = (40, 255, 255)
    edge_kernel_size: int = 5
    edge_min_contour_area_ratio: float = 0.01
    edge_min_contour_area_floor: int = 500
    edge_margin_pixels: int = 10
    edge_save_crops: Optional[bool] = None  # None -> follow mode; otherwise force on/off
    show_mask_preview: bool = True

    # Legacy IMX/Hailo config (kept for completeness/compatibility)
    imx_model_path: str = "/home/sir/Kevin_Walter/model_rpk_files/network.rpk"
    priors_path: str = "/home/sir/Kevin_Walter/utilities/ssd_priors.npy"
    preview_size: Tuple[int, int] = (320, 320)
    score_thresh: float = 0.01
    nms_iou_thresh: float = 0.05
    variances: Tuple[float, float] = (0.1, 0.2)
    class_labels_json: str = "/home/sir/Kevin_Walter/pipeline/imagenet-simple-labels.json"

    hef_path: str = "/home/sir/Kevin_Walter/hailo/mobilenet_v1.hef"
    hailo_batch_size: int = 16
    hailo_num_workers: int = 4
    hailo_input_shape: Tuple[int, int, int] = (224, 224, 3)
    hailo_throughput_mode: str = "end_to_end"

    crop_thread_workers: int = 4
    batch_submit_thread_count: int = 1
    batch_max_size: int = 16
    queue_maxsize: int = 20
    run_time: float = 45.0

    show_preview: bool = True
    verbose: bool = False

    test_mode: bool = False
    test_frames: int = 200
    json_output: Optional[str] = None

    # Profiling configuration
    profile_enabled: bool = True
    profile_log_interval_sec: float = 5.0
    profile_dump_path: Optional[str] = None
    profile_max_samples: int = 10000
    profile_mode: str = "full"  # one of: "full", "crop_only", "hailo_only"

    # Queue/backpressure tuning
    crop_queue_capacity: int = 200
    crop_put_timeout: float = 0.01
    # Max time (ms) a crop waits in assembler before forced flush (safety cap)
    max_batch_delay_ms: float = 15.0
