"""Grab bag of useful utilities."""

from threading import Lock
from collections import deque
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


class LockedString:
    """String that can only be written with a lock."""

    def __init__(self) -> None:
        """Initialize the object."""
        self._lock = Lock()
        self._text: str | None = None

    @property
    def text(self) -> str | None:
        """Read the string."""
        return self._text

    @text.setter
    def text(self, text: str) -> None:
        with self._lock:
            self._text = text


class MovingAveragePoseFilter:
    """Moving average filter for a rigid pose."""

    def __init__(self, queue_size: int = 10) -> None:
        """
        Create the filter.

        Args:
        ----
            queue_size (int): number of most recent samples to average over.

        """
        if queue_size <= 0:
            raise ValueError("queue_size cannot be negative.")

        self._queue_size = queue_size
        self._pos_history: deque[np.ndarray] = deque(maxlen=queue_size)
        self._ori_history: deque[np.ndarray] = deque(maxlen=queue_size)
        self._lock = Lock()

    def add_pose_to_queue(self, pos: np.ndarray, ori: R) -> None:
        """
        Add a new pose sample to the filter.

        Args:
        ----
            pos (np.ndarray): position in world frame
            ori (R): Rotation object in world frame

        """
        p = np.asarray(pos, dtype=float).reshape(3)
        R_mat = ori.as_matrix()

        with self._lock:
            self._pos_history.append(p)
            self._ori_history.append(R_mat)

    def get_filtered_pose(
            self,
            fallback_pos: np.ndarray,
            fallback_ori: R
    ) -> Tuple[np.ndarray, R]:
        """
        Get the filtered pose.

        If there are no samples yet, returns the fallback pose unchanged.

        Args:
        ----
            fallback_pos (np.ndarray): position to return if no history is available
            fallback_ori (R): orientation to return if no history is available

        Returns:
        -------
            (pos_avg, ori_avg) (Tuple[np.ndarray, R]): averaged position and orientation

        """
        with self._lock:
            if not self._pos_history:
                return np.asarray(fallback_pos, dtype=float).reshape(3), fallback_ori

            # average position
            pos_avg = np.mean(np.stack(self._pos_history, axis=0), axis=0)

            # average rotation matrices via SVD of the sum
            R_sum = np.zeros((3, 3), dtype=float)
            for R_i in self._ori_history:
                R_sum += R_i

            U, _, Vt = np.linalg.svd(R_sum)
            R_avg = U @ Vt

            # ensure det = +1
            if np.linalg.det(R_avg) < 0:
                U[:, -1] *= -1
                R_avg = U @ Vt

            ori_avg = R.from_matrix(R_avg)

        return pos_avg, ori_avg
