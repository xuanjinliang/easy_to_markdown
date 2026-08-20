import importlib.metadata

try:
    version = importlib.metadata.version(__package__)
except importlib.metadata.PackageNotFoundError:
    version = "0.0.0"
