#!/usr/bin/env python3
import pcl
import filters as fil

def Harrison(array):
    cloud = pcl.PointCloud()
    cloud.from_list(array[:, [0, 1, 2]])
    detector = pcl.HarrisKeypoint3D(cloud)
    detector.set_NonMaxSupression(True)
    detector.set_Radius(0.01)
    xyz = detector.compute()
    return xyz


