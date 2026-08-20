import os
from pathlib import Path

BasePkgPath = Path(__file__).resolve().parent
ModelDir = os.path.join(BasePkgPath.parent, 'model')
PdfTempDir = os.path.join(BasePkgPath.parent, 'pdf_temp')
PromptDir = os.path.join(BasePkgPath.parent, 'prompt')
MDDir = os.path.join(BasePkgPath.parent, 'md_dir')

DISABLE_AUTO_LOGGING_CONFIG = (
        os.getenv("EASY_2_MD_DISABLE_AUTO_LOGGING_CONFIG", "0") == "1"
)

AllowedFileExt = ('pdf', 'png', 'jpg', 'jpeg', 'webp')
Unknown = 'unknown'
PageName = "page_"
