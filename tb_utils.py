# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


def _get_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return None
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def setup(log_dir, enabled=False):
    if not enabled:
        return None
    return _get_writer(log_dir)


def log(writer, data, step):
    if writer is None:
        return
    for key, value in data.items():
        writer.add_scalar(key, value, step)


def close(writer):
    if writer is None:
        return
    writer.close()
