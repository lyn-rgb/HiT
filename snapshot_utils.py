# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil


_DEFAULT_EXTS = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".ipynb",
    ".toml",
    ".sh"
}


def snapshot_code(root_dir, out_dir, exts=None):
    exts = _DEFAULT_EXTS if exts is None else set(exts)
    os.makedirs(out_dir, exist_ok=True)

    skip_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "results",
        "samples",
        "wandb",
    }

    root_dir = os.path.abspath(root_dir)
    out_dir = os.path.abspath(out_dir)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel_dir = os.path.relpath(dirpath, root_dir)
        parts = rel_dir.split(os.sep)
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        if out_dir.startswith(dirpath):
            continue
        if any(part in skip_dirs for part in parts):
            continue

        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() not in exts:
                continue
            src_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(src_path, root_dir)
            dst_path = os.path.join(out_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
