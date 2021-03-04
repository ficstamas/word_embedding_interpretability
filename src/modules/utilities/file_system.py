import os


def init_filesystem(path: str, required_folders=()):
    os.makedirs(path, exist_ok=True)
    if len(required_folders) == 0:
        return

    for folder in required_folders:
        os.makedirs(os.path.join(path, folder), exist_ok=True)