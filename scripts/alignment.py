#!/usr/bin/env python3
import pcl
import numpy as np
import math

def icp(np1, np2):
    source = pcl.PointCloud()
    source.from_list(np1[:, [0, 1, 2]])
    target = pcl.PointCloud()
    target.from_list(np2[:, [0, 1, 2]])

    icp = source.make_IterativeClosestPoint()
    converged, transf, estimate, fitness = icp.icp(source, target)

    return np.concatenate((estimate.to_array(), np1[:, [3]]), axis=1)


def cluster_extraction(xyz, tree, rgb):
    ec = xyz.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.002)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(2000)
    ec.set_SearchMethod(tree)
    clusters_ind = ec.Extract()
    boxes_without_rgb = []
    boxes_with_rgb = []
    i = 0
    for j, indices in enumerate(clusters_ind):
        box_without_rgb = []
        box_with_rgb = []
        for i, ind in enumerate(indices):
            box_without_rgb.append([xyz[ind][0], xyz[ind][1], xyz[ind][2]])
            box_with_rgb.append([xyz[ind][0], xyz[ind][1], xyz[ind][2], rgb[ind][3]])
        boxes_without_rgb.append(box_without_rgb)
        boxes_with_rgb.append(box_with_rgb)
    return [boxes_without_rgb, boxes_with_rgb]

def get_normal_and_mean (boxes, index, map):
    for i in index:
        box = boxes[i, :]
        box = box[box[:, 3] != 0]
        cloud = pcl.PointCloud()
        cloud.from_list(box[:, [0, 1, 2]])
        tree = cloud.make_kdtree()
        ne = cloud.make_NormalEstimation()
        ne.set_SearchMethod(tree)
        ne.set_RadiusSearch(0.2)
        normals = ne.compute()
        normals = normals.to_array()
        mean = np.mean(box[:, [0, 1, 2, 3]], axis=0)
        if map[i][8] == 0:
            map[i] = np.concatenate((mean, np.mean(normals[:, [0, 1, 2]], axis=0), np.array((2, 1))), axis=0)
        elif 0 < angle_between_vectors(normals[:, [0, 1, 2]], map[i][4:7]) <= 1:
            map[i][7] = map[i][7] * 2
        elif 0 < np.linalg.norm(map[i][:3]-mean) <= 1:
            map[i][7] = map[i][7] * 2
        else:
            map[i][7] = map[i][7] / 2
            if (map[i][7] < 0.2):
                map[i] = np.concatenate((mean,  np.mean(normals[:, [0, 1, 2]], axis=0), np.array((1, 1))), axis=0)
    return None

def angle_between_vectors(v1, v2):
    unit_v1 = v1/np.linalg.norm(v1)
    unit_v2 = v2/np.linalg.norm(v2)
    dot_product = np.dot(unit_v2, unit_v1)
    return math.degrees(np.arccos(dot_product))
