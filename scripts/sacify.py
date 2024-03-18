#!/usr/bin/env python3
import ros_numpy
import numpy as np
import pcl

cloud = pcl.PointCloud_PointXYZRGB()

def ransac(cloud):
    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_CIRCLE3D)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.01)
    seg.set_normal_distance_weight(0.01)
    seg.set_max_iterations(100)
    [indices, coefficients] = seg.segment()
    return indices
