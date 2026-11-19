import argparse
import yaml, os
from pathlib import Path


def load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}

    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config(args: argparse.Namespace) -> dict:
    config = load_config(args.config)

    # PDF
    pdf_config = config.setdefault("pdf", {})
    pdf_config["dpi"] = args.pdf_dpi if args.pdf_dpi is not None else pdf_config.get("dpi", 200)
    pdf_config["img_suffix"] = (
        args.pdf_img_suffix
        if args.pdf_img_suffix is not None
        else pdf_config.get("img_suffix", "webp")
    )

    llm_config = config.setdefault("llm", {})

    for key in ("model", "temperature", "base_url", "api_key"):
        value = getattr(args, key, None)

        if value is not None:
            llm_config[key] = value

    api_key = (
            args.api_key
            or os.getenv("LLM_API_KEY")
    )

    if api_key is not None:
        llm_config.setdefault("api_key", api_key)

    config["input"] = args.input
    config["output"] = args.output
    config["temp_dir"] = args.temp_dir

    return config
