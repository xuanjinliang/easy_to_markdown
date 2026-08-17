import unittest
from generate.download_model import download_model


class TestDownloadModel(unittest.TestCase):
    def test_download(self):
        download_model(
            repo_id="PaddlePaddle/PP-DocLayoutV3",
            filename="paddle_test/PP-DocLayoutV3",
        )
