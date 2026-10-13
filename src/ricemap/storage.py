"""
Functions to read and write files for the whole project
Flexible between local disk or in Google Cloud Storage, based on the environment variable
    STORAGE_URI which sets the root folder (default: ./outputs).
"""

import os
import posixpath
from typing import IO

import fsspec


def uri(*parts: str) -> str:
    # Make a path from the STORAGE_URI variable and the given parts.
    # If STORAGE_URI is not set, use ./outputs.
    base = os.environ.get("STORAGE_URI", "./outputs").rstrip("/")
    return "/".join([base, *parts])


def exists(path: str) -> bool:
    # Return True if a file exists at the path. The path is local or gs://.
    fs, fs_path = fsspec.core.url_to_fs(path)
    return fs.exists(fs_path)


def open_file(path: str, mode: str = "rb") -> IO:
    # Open a file for read or write. In write mode, make the parent folders first.
    fs, fs_path = fsspec.core.url_to_fs(path)
    if "w" in mode:
        fs.makedirs(posixpath.dirname(fs_path), exist_ok=True)
    return fs.open(fs_path, mode)


def upload(local_path: str, dest: str) -> None:
    # Copy a local file to the destination. Make the parent folders first.
    fs, fs_path = fsspec.core.url_to_fs(dest)
    fs.makedirs(posixpath.dirname(fs_path), exist_ok=True)
    fs.put_file(local_path, fs_path)

def list_files(pattern: str) -> list[str]:
    # Return the full paths that match the pattern, for example uri("tiles", "*.tif").
    fs, fs_path = fsspec.core.url_to_fs(pattern)
    return sorted(fs.unstrip_protocol(p) for p in fs.glob(fs_path))

