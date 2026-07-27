import pkg
import hashlib

def get_file(file_path: str) -> bytes:
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return file_bytes

def get_file_extension(file_bytes: bytes) -> str:
    signatures = {
        b'%PDF-': 'pdf'
    }

    # Determine the file extension
    for signature, extension in signatures.items():
        if file_bytes.startswith(signature):
            return extension

    return 'unknown'


def img_allowed_file(filename: str) -> bool:
    return '.' in filename and filename.split('.')[-1].lower() in pkg.AllowedFileExt

def file_sha256(path, chunk_size=8192):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()