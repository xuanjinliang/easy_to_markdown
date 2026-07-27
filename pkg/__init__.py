import os
from pathlib import Path

ModelDir = os.path.join(os.getcwd(), 'model')
PdfTempDir = os.path.join(os.getcwd(), 'pdf_temp')

BatchJsonDir = os.path.join(PdfTempDir, 'batch_json')

BasePkgPath = Path(__file__).resolve().parent
PromptDir = os.path.join(BasePkgPath.parent, 'prompt')

AllowedFileExt = ('pdf', 'png', 'jpg', 'jpeg', 'webp')
Unknown='unknown'
PageName="page_"