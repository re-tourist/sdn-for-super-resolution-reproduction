"""Dataset helpers for the repo's staged reproduction workflow."""

from src.datasets.stage4_emnist import (
    DEFAULT_STAGE4_DATASET_ROOT,
    Stage4EMNISTDataset,
    build_stage4_train_val_datasets,
    resolve_stage4_dataset_root,
)
from src.datasets.stage5_emnist_display import (
    DEFAULT_STAGE5_CANVAS_HW,
    DEFAULT_STAGE5_CELL_HW,
    DEFAULT_STAGE5_DATASET_ROOT,
    DEFAULT_STAGE5_GRID_SHAPE,
    DEFAULT_STAGE5_LETTER_COUNT_CHOICES,
    DEFAULT_STAGE5_SAMPLE_COUNTS,
    Stage5AugmentationConfig,
    Stage5EMNISTDisplayDataset,
    build_stage5_emnist_display_dataset,
)
from src.datasets.target_adapter import (
    Stage4RoiTargetAdapter,
    save_stage4_input_target_preview,
)

__all__ = [
    "DEFAULT_STAGE4_DATASET_ROOT",
    "DEFAULT_STAGE5_CANVAS_HW",
    "DEFAULT_STAGE5_CELL_HW",
    "DEFAULT_STAGE5_DATASET_ROOT",
    "DEFAULT_STAGE5_GRID_SHAPE",
    "DEFAULT_STAGE5_LETTER_COUNT_CHOICES",
    "DEFAULT_STAGE5_SAMPLE_COUNTS",
    "Stage4EMNISTDataset",
    "Stage4RoiTargetAdapter",
    "Stage5AugmentationConfig",
    "Stage5EMNISTDisplayDataset",
    "build_stage4_train_val_datasets",
    "build_stage5_emnist_display_dataset",
    "resolve_stage4_dataset_root",
    "save_stage4_input_target_preview",
]
