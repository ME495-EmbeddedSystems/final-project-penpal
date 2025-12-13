"""Globally defined useful constants."""

from scipy.spatial.transform import Rotation as R
import numpy as np


T_EE_pen = np.array(
    [
        [0, 0, -1, 0.08],
        [0, 1, 0, 0],
        [1, 0, 0, 0.02],
        [0, 0, 0, 1],
    ]
)
"""
Pen tip expressed in TCP frame

orientation:
- pen x is ee -z
- pen y is ee Y
- pen z is ee x

"""


R_tcp_board = R.from_matrix(
    np.array(
        [
            [0, 0, -1],
            [1, 0, 0],
            [0, -1, 0],
        ]
    )
)

R_board_tcp = R_tcp_board.inv()

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

board +x is TCP +y
board +y is TCP -z
board +z is TCP -x

"""
