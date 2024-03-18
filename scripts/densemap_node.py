#!/usr/bin/env python3
import rospy
import message_filters
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped
from tf.transformations import *
import queue
import ros_numpy
import numpy as np
import pcl
from dynamic_reconfigure.server import Server
from mediassist3_densemap.cfg import dense_map_nodeConfig

# custom modules
import prefilter
import sacify as sac
import alignment as align
import filters as fil
import voxel_grid as vox


class DenseMap():

    def __init__(self):

        # Make this a ROS node:
        rospy.init_node("densemap_python")
        rospy.loginfo("Node initialized")

        # Listen for messages containing the camera pose:
        self.subCameraPose = message_filters.Subscriber("orb_slam2_stereo/pose",
                                                        PoseStamped)

        # Listen for messages containing the 3D reconstruction of the current frame:
        self.subPointCloud = message_filters.Subscriber("request/stereo/points2",
                                                        PointCloud2)

        # Listen for segmentations of the current frame:
        self.subSegmentation = message_filters.Subscriber("request/image_segmented",
                                                          Image)

        # Synchronize the subscribers:
        self.ts = message_filters.TimeSynchronizer(
            [self.subCameraPose, self.subPointCloud, self.subSegmentation],
            1000)
        self.ts.registerCallback(self.parseData)
        rospy.loginfo("Created subscribers")

        # Set up publisher which publishes the full dense map as a point cloud:
        self.pub = rospy.Publisher("~/points2", PointCloud2, queue_size=1)
        rospy.loginfo("Created publisher")

        # Read configuration from parameters:
        self.inputInMillimeters = rospy.get_param("input_in_mm", False)
        self.cutoffTop = rospy.get_param("image_cutoff_top", 0)
        self.cutoffBottom = rospy.get_param("image_cutoff_bottom", 0)
        self.cutoffLeft = rospy.get_param("image_cutoff_left", 0)
        self.cutoffRight = rospy.get_param("image_cutoff_right", 0)

        rospy.loginfo("Configuration:\n" +
                      "\tInput in Millimeters: %s\n" +
                      "\tBorder to ignore:\n" +
                      "\t\tTop: %s\n" +
                      "\t\tBottom: %s\n" +
                      "\t\tLeft: %s\n" +
                      "\t\tRight: %s",
                      self.inputInMillimeters,
                      self.cutoffTop,
                      self.cutoffBottom,
                      self.cutoffLeft,
                      self.cutoffRight)

        self.map = None
        self.target = None
        self.kdtree = None
        self.queue = queue.Queue(3)

        # dynamic reconfigure parameters
        self.prefilter = None
        self.prefilter_segmentation = None
        self.prefilter_color = None
        self.prefilter_distance = None
        self.segmentation_count = None
        self.hsv_value = None
        self.dist_min = None
        self.dist_max = None
        self.statistical_outlier = None
        self.voxel_grid_filter = None
        self.approximate_voxel_grid_filter = None
        self.fullshuffle = None

    def parseData(self, cameraPoseMsg, pointCloudMsgIn, segmentationMsg):
        rospy.loginfo("-" * 40)
        rospy.loginfo("Data received")

        # Convert point cloud to numpy array:
        arr = ros_numpy.point_cloud2.pointcloud2_to_array(pointCloudMsgIn)
        rospy.loginfo("Incoming point cloud shape: %s", arr.shape)

        # Cut off (black) borders:
        top = self.cutoffTop
        bottom = arr.shape[0] - self.cutoffBottom
        left = self.cutoffLeft
        right = arr.shape[1] - self.cutoffRight
        arr = arr[top:bottom, left:right]
        rospy.loginfo("Shape after discarding borders: %s", arr.shape)

        # Ignore shape (because from this point on we'll not be working with the
        # cloud as a depth image):
        arr = arr.flatten()

        rospy.loginfo("Flattened: %s", arr.shape)

        #####
        # subsampling / downsampling with a random choice of values
        #####

        # convert segmentationMassage to ndarray and bring it in the same shape as the points
        # segmentationMsg = <class 'sensor_msgs.msg._Image.Image'>

        segmentationArray = ros_numpy.image.image_to_numpy(segmentationMsg)
        segmentationArray = segmentationArray.flatten()

        # take random indices (10% of incomimg values)
        randomInices = np.random.choice(len(arr), round(len(arr) / 10), replace=False)

        arrList = []
        segmentationList = []

        for i in randomInices:
            arrList.append(arr[i])
            segmentationList.append(segmentationArray[i])

        arr = np.array(arrList)
        segmentationArray = np.array(segmentationList)

        rospy.loginfo("Downsampled cloud shape: %s", arr.shape)

        #########################################################################

        # Find the number of points which we've received:
        N = arr.shape[0]
        rospy.loginfo("Received points: %s", N)
        rospy.loginfo("Field names: %s", arr.dtype.names)
        rospy.loginfo("Field formats: %s", arr.dtype)

        # Extract the position and add a "1" to each point to turn into
        # homogeneous coordinates
        xyz = np.stack([arr["x"], arr["y"], arr["z"], np.ones(N)], axis=1)

        # "<f4" = float32 after np.stack dtype changes to foat64
        rospy.loginfo("Field formats after np.stack: %s", xyz.dtype)

        # To transform from camera coordinates to world coordinates, rotate
        # by 90 degrees around Z and Y axes:normlas2
        qz = quaternion_about_axis(-math.pi * 0.5, (0, 0, 1))
        qy = quaternion_about_axis(math.pi * 0.5, (0, 1, 0))
        qCamToWorld = quaternion_multiply(qy, qz)
        # Additionally, rotate by the rotation as given in the camera pose:
        qCam = [cameraPoseMsg.pose.orientation.x,
                cameraPoseMsg.pose.orientation.y,
                cameraPoseMsg.pose.orientation.z,
                cameraPoseMsg.pose.orientation.w]

        q = quaternion_multiply(qCam, qCamToWorld)

        # Turn into rotation matrix:
        rotMatrix = quaternion_matrix(q)
        scaleMatrix = scale_matrix(1)
        if self.inputInMillimeters:
            scaleMatrix = scale_matrix(0.001)

        # Join scale and rotation:
        transformMatrix = np.dot(rotMatrix, scaleMatrix)

        # Add camera position:
        transformMatrix[0:3, 3] = [cameraPoseMsg.pose.position.x,
                                   cameraPoseMsg.pose.position.y,
                                   cameraPoseMsg.pose.position.z]
        rospy.loginfo("Input transformation matrix:\n%s", transformMatrix)

        # Apply transformation matrix to the points:
        points = np.dot(transformMatrix, xyz.T).T

        # Add the color field to the list of points:
        points[:, 3] = arr["rgb"]

        ############################# prefilter #############################

        # points = ndarray

        # 1. segmentaton filter
        # 2. color filter
        # 3. distance filter

        # enable/diable all prefilters
        if self.prefilter:
            rospy.loginfo("prefilters active")

            #####
            # segmentation filter
            #####

            # delete some of the points which the segmentation determend as non-liver
            # because we don't need that much information about the surrounding

            # enable/diable segmatation filter
            if self.prefilter_segmentation:
                rospy.loginfo("segmentation filter active")

                # find non-liverpoints via segmentetion
                non_liver_indices = prefilter.find_segmentation_indices(segmentationArray, self.segmentation_count)

                # delete every n-th (curred every 3rd) non-liverpoint
                points_list = points.tolist()
                points_list = prefilter.delete_from_indices(points_list, non_liver_indices)

                points = np.array(points_list)

            #####
            # color filter
            #####

            # delete points which are too bright (default hsv value >= 0.9)

            # enable/diable color filter
            if self.prefilter_color:
                rospy.loginfo("color filter active")

                # dtype = float64
                color_indices = prefilter.find_color_indices(points[:, 3], self.hsv_value)

                points_list = points.tolist()
                points_list = prefilter.delete_from_indices(points_list, color_indices)

                points = np.array(points_list)

            #####
            # point-cameraPose distance filter (default threshold: 8 - 28 cm)
            #####

            # delete all points which are not in a certin distance to camera pose

            # enable/diable distance filter
            if self.prefilter_distance:
                rospy.loginfo("cam dist filter active")

                # current position of the camera in the scene
                cam_pose = (cameraPoseMsg.pose.position.x, cameraPoseMsg.pose.position.y, cameraPoseMsg.pose.position.z)

                points = prefilter.dist_point_cam_filter(points, cam_pose, self.dist_min, self.dist_max)
        
        #############################################################################
        # we call the input coordinates 'pointcloud'                                #
        # voxelGrid boxes from 'pointcloud' are called 'cloud'                      #
        # voxelGrid boxes from result of NormalExtraction est... is called 'map'    #
        #############################################################################

        # map structure is (x,y,z,rgb, x_n, y_n, z_n, e, b)
        # point coordinates x,y,z
        # colour value rgb
        # the normal x_n, y_n, z_n
        # error value e
        # check value b
        self.map = np.zeros((int(200 ** 3), 9), dtype='<f8')

        # create pcl pointcloud
        pointcloud = pcl.PointCloud_PointXYZRGB()
        pointcloud.from_list(points)

        # passtrough Filter, don't change it
        pointcloud = fil.passtroughfilter(pointcloud)

        # enable/disable statistical outlier filter
        if self.statistical_outlier:
            # Filter for statistical outliers (x)
            pointcloud = fil.statistical_outlier_filter_RGB(pointcloud)

        # [INFO]
        # enable only one at the same time
        # voxel_grid_filter OR approximate_voxel_grid_filter

        # enable/disable voxel grid filter
        if self.voxel_grid_filter:
            # Compress pointcloud app (x)
            pointcloud = fil.voxel_grid_filter(pointcloud)

        # enable/disable approximate voxel grid filter
        if self.approximate_voxel_grid_filter:
            pointcloud = fil.approximate_voxel_grid_filter(pointcloud)

        if not self.queue.full():
            # making clouds of any pointcloud + index
            cloud, index = vox.voxel_boxes(pointcloud)

            # edit map with given cloud
            align.get_normal_and_mean(cloud, index, self.map)

            # store clouds and indexes in Queue
            self.queue.put([cloud, index])

        if self.queue.full() and self.target is None:

            # mask out every valid point by e and b Value from map
            securepoints = self.map[(self.map[:, 8] == 1) & (self.map[:, 7] > 1)]
            
            # calculate index with given securepoints
            index = np.apply_along_axis(indexer, 1, securepoints[:, [0, 1, 2]]).T
            cloud1, i1 = self.queue.get()
            cloud2, i2 = self.queue.get()
            cloud3, i3 = self.queue.get()

            # get the intersection between index from securepoints and clouds
            index1 = np.intersect1d(index, i1)
            index2 = np.intersect1d(index, i2)
            index3 = np.intersect1d(index, i3)

            self.target = np.array([(0,0,0,0)])
            # self.map = securepoints[:, [0,1,2,3]])


            for box in cloud1[index1, :]:
                self.target = np.concatenate((self.target, box))

            for box in cloud2[np.setdiff1d(index2, index1), :]:
                self.target = np.concatenate((self.target, box))

            for box in cloud3[np.setdiff1d(index3, index2), :]:
                self.target = np.concatenate((self.target, box))

            # We should do some variables for the clock System
            # If we want a full restore of clouds, like load 2/kill 2   -> fullshuffle = True
            # If we want a small restore, like load 1/kill 1            -> fullshuffle = False
            if self.fullshuffle:
                self.queue.put([cloud2, i2])
            else:
                self.queue.put([cloud2, i2])
                self.queue.put([cloud3, i3])

        elif self.queue.full():

            securepoints = self.map[(self.map[:, 8] == 1) & (self.map[:, 7] > 1)]
            index = np.apply_along_axis(indexer, 1, securepoints[:, [0, 1, 2]]).T
            cloud1, i1 = self.queue.get()
            cloud2, i2 = self.queue.get()
            cloud3, i3 = self.queue.get()

            index1 = np.intersect1d(index, i1)
            index2 = np.intersect1d(index, i2)
            index3 = np.intersect1d(index, i3)

            if self.fullshuffle:
                for box in cloud2[np.setdiff1d(index2, index1), :]:
                    self.target = np.concatenate((self.target, box))

                for box in cloud3[np.setdiff1d(index3, index2), :]:
                    self.target = np.concatenate((self.target, box))

                self.queue.put([cloud2, i2])
            else:
                for box in cloud1[index1, :]:
                    self.target = np.concatenate((self.target, box))

                self.queue.put([cloud2, i2])
                self.queue.put([cloud3, i3])

        # [INFO]
        # some duplicates exist but not that many, maybe another approximate voxel grid filter could help

        # Convert the points back to a structured array:
        if self.target is not None:
            # for faster publish, comment if you want original
            wolkig = pcl.PointCloud_PointXYZRGB()
            wolkig.from_list(self.target[:, [0,1,2,3]])
            filter = pcl.ApproximateVoxelGrid_PointXYZRGB()
            filter.set_InputCloud(wolkig)
            filter.set_leaf_size(0.00001, 0.00001, 0.00001)
            wolkig = filter.filter()
            pcl.save( wolkig, "wolke7.pcd")

            structuredArray = np.core.records.fromarrays(
                wolkig.to_array().transpose(),
                names='x, y, z, rgb',
                formats='f4, f4, f4, f4')

            rospy.loginfo("Publishing map conatining %s points",
                          structuredArray.shape[0])

            # Convert the structured array to a point cloud
            pointCloudMsgOut = ros_numpy.point_cloud2.array_to_pointcloud2(
                structuredArray,
                stamp=pointCloudMsgIn.header.stamp,
                frame_id="map")

            # Send the point cloud on to any subscribers
            self.pub.publish(pointCloudMsgOut)

# Index function
def indexer(x):
    box_size = 0.2  # cm
    axis_length = 40  # cm
    num_boxes_per_axis = axis_length / box_size
    offset = num_boxes_per_axis / 2

    gx = x[0] * 100 // box_size + offset
    gy = x[1] * 100 // box_size + offset
    gz = x[2] * 100 // box_size + offset

    return int(gx + gy * num_boxes_per_axis + num_boxes_per_axis ** 2 * gz)

# method to update parameters with ros dynamic reconfigure
def callback(config, level):
    rospy.loginfo("Reconfigure is active")
    rospy.loginfo(f"Config parameters: {config}")

    # map config parameters to node parameters
    denseMap.prefilter = config.prefilter
    denseMap.prefilter_segmentation = config.prefilter_segmentation
    denseMap.prefilter_color = config.prefilter_color
    denseMap.prefilter_distance = config.prefilter_distance
    denseMap.segmentation_count = config.segmentation_count
    denseMap.hsv_value = config.hsv_value
    denseMap.dist_min = config.dist_min
    denseMap.dist_max = config.dist_max
    denseMap.statistical_outlier = config.statistical_outlier
    denseMap.voxel_grid_filter = config.voxel_grid_filter
    denseMap.approximate_voxel_grid_filter = config.approximate_voxel_grid_filter
    denseMap.fullshuffle = config.fullshuffle

    return config


if __name__ == '__main__':
    denseMap = DenseMap()

    # Server for dynamic reconfigure
    srv = Server(dense_map_nodeConfig, callback)

    # Continue receiving messages until node shutdown:
    rospy.spin()



