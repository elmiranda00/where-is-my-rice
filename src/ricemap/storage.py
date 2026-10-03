import os
import posixpath
from typing import IO

import fsspec


def uri(*parts: str) -> str:
    base = os.environ.get("STORAGE_URI", "./outputs").rstrip("/")
    return "/".join([base, *parts])


def exists(path: str) -> bool:
    fs, fs_path = fsspec.core.url_to_fs(path)
    return fs.exists(fs_path)


def open_file(path: str, mode: str = "rb") -> IO:
    fs, fs_path = fsspec.core.url_to_fs(path)
    if "w" in mode:
        fs.makedirs(posixpath.dirname(fs_path), exist_ok=True)
    return fs.open(fs_path, mode)


def upload(local_path: str, dest: str) -> None:
    fs, fs_path = fsspec.core.url_to_fs(dest)
    fs.makedirs(posixpath.dirname(fs_path), exist_ok=True)
    fs.put_file(local_path, fs_path)
