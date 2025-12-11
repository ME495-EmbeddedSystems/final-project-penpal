"""Globally defined useful constants."""

from scipy.spatial.transform import Rotation as R
import numpy as np


# R_board_tcp = R.from_matrix(
#     np.array(
#         [
#             [0, 0, -1],
#             [-1, 0, 0],
#             [0, 1, 0],
#         ]
#     )
# )
R_board_tcp = R.from_matrix(
    np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
)
"""
Rotation for the transform from board frame to TCP frame.

board frame:
+z points away from board normal to board
+y is upwards
+x is to the right

TCP frame:
z points straight out from the gripper
y points towards the black button
x comes out at you when black button 
  is counterclockwise of the forearm link. 

board +z is TCP +x
board +y is TCP +z
board +x is TCP +y

"""
