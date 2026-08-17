from transformers.processing_utils import transformers_module

import pkg
import os
from pathlib import Path
from huggingface_hub import snapshot_download
import logging
from logging import NullHandler

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


def download_model(repo_id: str, filename: str):
    output_dir = Path(os.path.join(pkg.ModelDir, filename))
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(
        repo_id=repo_id,
        local_dir=os.path.join(pkg.ModelDir, filename),
    )

    logger.info(f"{repo_id} --> {str(model_path)}")


def request_model_config() -> list[dict[str, str]]:
    config = [
        {
            "repo_id": "PaddlePaddle/PP-DocLayoutV3",
            "filename": "paddle/PP-DocLayoutV3",
        },
        {
            "repo_id": "PaddlePaddle/PaddleOCR-VL-1.6",
            "filename": "paddle/PP-DocLayoutV3",
        },
        {
            "repo_id": "PaddlePaddle/PP-LCNet_x1_0_table_cls",
            "filename": "paddle/PP-LCNet_x1_0_table_cls",
        },
        {
            "repo_id": "PaddlePaddle/PP-LCNet_x1_0_textline_ori",
            "filename": "paddle/PP-LCNet_x1_0_textline_ori",
        },
        {
            "repo_id": "PaddlePaddle/PP-OCRv6_medium_det",
            "filename": "paddle/PP-OCRv6_medium_det",
        },
        {
            "repo_id": "PaddlePaddle/PP-OCRv6_medium_rec",
            "filename": "paddle/PP-OCRv6_medium_rec",
        },
        {
            "repo_id": "PaddlePaddle/RT-DETR-L_wired_table_cell_det",
            "filename": "paddle/RT-DETR-L_wired_table_cell_det",
        },
        {
            "repo_id": "PaddlePaddle/RT-DETR-L_wireless_table_cell_det",
            "filename": "paddle/RT-DETR-L_wireless_table_cell_det",
        }
    ]

    return config


def optional_model_config() -> dict[str, list[dict[str, str]]]:
    config = {
        "transformers": [
            {
                "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
                "filename": "qwen/Qwen3-VL-4B-Instruct",
            }
        ],
        "mlx": [
            {
                "repo_id": "mlx-community/Qwen3-VL-4B-Instruct-8bit",
                "filename": "qwen_mlx/Qwen3-VL-4B-Instruct-8bit",
            }
        ]
    }
    return config
