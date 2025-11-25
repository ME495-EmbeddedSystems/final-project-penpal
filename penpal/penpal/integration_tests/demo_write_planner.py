"""Demo write planner to write demo trajectory."""

import numpy as np

from scipy.spatial.transform import Rotation as R

from penpal.control.pp_control import Trajectory


class DemoWritePlanner:
    """Compute trajectories to write on the real board."""

    def __init__(
        self,
        board_center : np.ndarray,
        board_orientation : np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
    ) -> None:
        """
        Initialize the object.

        Args:
            board_center: [x,y,z]
            board_orientation: [qx,qy,qz,qw]
        """
        self.board_width = 0.7  # in meters
        self.board_height = 0.5  # in meters
        self.x = 0
        self.y = 0
        self.line_height = 0.1
        self.line_space = 0.01
        self.center = board_center
        self.orientation = board_orientation
        self.rotation = R.from_quat(board_orientation)

    def write_characters(self, characters) -> None:
        """
        Write a list of characters to the board.

        Creates newlines when necessary.

        Args:
            characters (list[Character]): list of characters to write.
            [x, y, z, fx, fy, fz]

        """
        final_traj = []
        for i, char in enumerate(characters):
            points = char.data
            x_coords = points[:, 0]
            char_width = np.max(x_coords) - np.min(x_coords)
            x_offset = - np.min(x_coords)
            if char_width > self.board_width - self.x:
                self.y += self.line_height
                self.x = 0

            new_point = []
            for point in points:
                local_x = point[0] + self.x + x_offset
                local_y = point[1] + self.y
                local_z = point[2]

                local_vector = np.array([local_x, local_y, local_z])
                offset = self.rotation.apply(local_vector)

                new_x = self.center[0] + offset[0]
                new_y = self.center[1] + offset[1]
                new_z = self.center[2] + offset[2]

                new_point.append([new_x, new_y, new_z,
                                  point[3], point[4], point[5]])

            label = f'{char.label}_{i}'
            new_traj = Trajectory(label, np.array(new_point))
            final_traj.append(new_traj)
            self.x += char_width + self.line_space

        return final_traj
