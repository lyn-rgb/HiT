# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys


class TrainingLogger:
    def __init__(self, log_dir, rank=0, name="train", log_all_ranks=False):
        self._logger = logging.getLogger(f"{name}.{rank}")
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)

        if self._logger.handlers:
            return

        formatter = logging.Formatter(
            fmt="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if log_all_ranks or rank == 0:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            self._logger.addHandler(stream_handler)

        if rank == 0:
            if log_dir is None:
                raise ValueError("log_dir is required for rank 0 logger")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "log.txt")
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def error(self, msg):
        self._logger.error(msg)
