# voxel grid methods for densemap
import numpy as np


def voxel_boxes(points):
    # Cloud structure (x,y,z,rgb)
    boxes = np.zeros((int(200 ** 3), 30, 4), dtype='<f8')

    # Indexes from changed Clouds
    index = np.unique(np.apply_along_axis(index_function, 1, points.to_array(), boxes).T)

    return boxes, index


def get_box_fill_level(box):
    fill_level = 0
    for i in box:
        if i[0] == 0:
            break
        fill_level += 1

    return fill_level



def index_function(x, boxes):
    # Grid parameters
    box_size = 0.2  # cm
    axis_length = 40  # cm
    num_boxes_per_axis = axis_length / box_size
    points_per_box = 30
    offset = num_boxes_per_axis / 2

    gx = x[0] * 100 // box_size + offset 
    gy = x[1] * 100 // box_size + offset
    gz = x[2] * 100 // box_size + offset

    # Index function for Cloud
    index = int(gx + gy*num_boxes_per_axis + num_boxes_per_axis**2 * gz)

    next_free_spot = None
    for position in range(points_per_box):
        if boxes[index, position, 0] == 0:
            next_free_spot = position
            break

    # if free spot is found add point
    if next_free_spot is not None:
        boxes[index, next_free_spot, :] = x
    return index