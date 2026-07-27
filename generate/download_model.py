import pkg
import os
from huggingface_hub import hf_hub_download

def download_model(repo_id: str, filename: str):
    os.makedirs(pkg.ModelDir, exist_ok=True)

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=os.path.join(pkg.ModelDir, "cache")
    )

    print(model_path)