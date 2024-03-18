#!/usr/bin/env python3
import pcl
import numpy as np
import queue


# filters very strong
def statistical_outlier_filter_RGB(rgb):
    """ Filters given pc by calculating the mean distance between of K points, multiplied by an other threshold value

        Args: rgb (pcl.PointCloud_PointXYZRGB)

        Returns: pcl.PointCloud_PointXYZRGB
    """
    fil = pcl.StatisticalOutlierRemovalFilter_PointXYZRGB(rgb)
    fil.set_mean_k(50)
    fil.set_std_dev_mul_thresh(1.0)
    return fil.filter()
# Features are needed to do a better fusion
# Triangolate Cloud and do Cross multiplications for Normals



def approximate_voxel_grid_filter(rgb):
    """ Convert pc into a grid structure, merch near points

            Args: rgb (pcl.PointCloud_PointXYZRGB)

            Returns: pcl.PointCloud_PointXYZRGB
    """

    fil = pcl.ApproximateVoxelGrid_PointXYZRGB()
    fil.set_InputCloud(rgb)
    fil .set_leaf_size(0.0045, 0.0045, 0.0045)
    return fil.filter()

def voxel_grid_filter(rgb):
    """ Convert pc into a grid structure, merch near points

            Args: rgb (pcl.PointCloud_PointXYZRGB)

            Returns: pcl.PointCloud_PointXYZRGB
    """

    fil = rgb.make_voxel_grid_filter()
    fil .set_leaf_size(0.0045, 0.0045, 0.0045)
    return fil.filter()

def moving_least_square(rgb):
    mls = pcl.MovingLeastSquares_PointXYZRGB(rgb)
    mls.set_polynomial_fit(True)
    mls.set_polynomial_order(2)
    mls.set_search_radius(0.03)
    return mls.process()

def conditional_removing(rgb):
    """ Filter pc by condition (GT, GE, LT, LE, EQ)

            Args: rgb (pcl.PointCloud_PointXYZRGB)

            Returns: pcl.PointCloud
    """

    rgb.to_arary()
    cloud = pcl.PointCloud()
    cloud.from_list(rgb.to_arary()[:, [0, 1, 2]])
    cond = pcl.ConditionAnd(cloud)
    for i in ['x', 'y', 'z']:
        cond.add_Comparison2(i, pcl.CythonCompareOp_Type.GT, np.NINF)


    condrm = pcl.ConditionalRemoval(cloud)
    condrm.set_Condition(cond)
    condrm.set_KeepOrganized(True)

    return condrm.filter()

def passtroughfilter(xyz):
    """ Filter pc by limits

            Args: rgb (pcl.PointCloud())

            Returns: pcl.PointCloud_PointXYZRGB
    """
    for i in ["x", "y", "z", "rgb"]:
       passthrough = xyz.make_passthrough_filter()
       passthrough.set_filter_field_name(i)
       passthrough.set_filter_limits(np.NINF, np.Inf)
       return passthrough.filter()