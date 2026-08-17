import unittest
import pkg
from generate.download_model import download_model


class TestDownloadModel(unittest.TestCase):
    def test_download(self):
        download_model(
            repo_id="PaddlePaddle/PP-DocLayoutV3",
            filename="paddle_test/PP-DocLayoutV3",
        )

    def test_pkg_dir(self):
        print(f"model_dir --> {pkg.ModelDir}")
        print(f"pdf_temp_dir --> {pkg.PdfTempDir}")
        print(f"prompt_dir --> {pkg.PromptDir}")