# Whiteboard pose, visiblity, and configuration info

geometry_msgs/PoseStamped pose
# 6d pose of the top left corner of the whiteboard

float32 width_m
float32 height_m
# width & height of the whiteboard

float32[] writeable_area
# Rectangular region available for writing, relative to the board origin.
# [x_tl, y_tl, x_br, y_br].
# Note that board coordinates are in R2 with +x = right, -y = down

int8 n_tags
# number of AprilTags on which this pose estimate is based

int32 sequence_number
# sequence number indicating number of continuous valid pose estimates.
# resets to 0 if a board pose cannot be generated.