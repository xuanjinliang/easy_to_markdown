import hashlib
from pkg.files_handle import get_file


# read the image from file for get a hash code to seed
def image_to_seed(image_file: str) -> int:
    image_bytes = get_file(image_file)
    hash_hex = hashlib.sha256(image_bytes).hexdigest()
    seed = int(hash_hex, 16) % (2 ** 31 - 1)

    return seed


def str_to_seed(data: str) -> int:
    return int(hashlib.md5(data.encode()).hexdigest(), 16) % (2 ** 31 - 1)


def get_image_extension(image_bytes: bytes) -> str:
    # Dictionary of image file signatures
    signatures = {
        b'\xFF\xD8\xFF': 'jpeg',
        b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A': 'png',
    }

    # Determine the file extension
    for signature, extension in signatures.items():
        if image_bytes.startswith(signature):
            return extension

    if len(image_bytes) >= 12 and image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        return 'webp'

    return 'unknown'
