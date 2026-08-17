import os
from pathlib import Path

BasePkgPath = Path(__file__).resolve().parent
ModelDir = os.path.join(BasePkgPath.parent, 'model')
PdfTempDir = os.path.join(BasePkgPath.parent, 'pdf_temp')
PromptDir = os.path.join(BasePkgPath.parent, 'prompt')

AllowedFileExt = ('pdf', 'png', 'jpg', 'jpeg', 'webp')
Unknown='unknown'
PageName="page_"