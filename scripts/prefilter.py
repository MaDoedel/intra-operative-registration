# prefilter methods for densemap

import math
import colorsys

# square distance between 2 points in 3D
def dist_point_point(x1, y1, z1, x2, y2, z2):
	return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)



# keeps only points within a specified range to the Camera
# points = nd array with the current points to be filtered
# cam_pose = 3-Tuple with coordinates of the camera
def dist_point_cam_filter(points, cam_pose, dist_min, dist_max):
	dist_point_cam_filtered = []
	for point in points:
		dist_point_cam = dist_point_point(point[0], point[1], point[2], cam_pose[0], cam_pose[1], cam_pose[2])
		# keep only points within a certin distance window from camPose 
		if (dist_point_cam < dist_max and dist_point_cam > dist_min):
			dist_point_cam_filtered.append(point)

	return dist_point_cam_filtered



# find every non-liver point via segmentation message and take the indices to delete some of them
# segmentaton = ndarray
# segmentation_count -> every n'th non-liver point will be deleted
# returns list of indices(int)
def find_segmentation_indices(segmentation, segmentation_count):
	non_liver_count = 0
	indices = []
	for i in range(len(segmentation)):
		if segmentation[i] != 0:
			continue
		
		non_liver_count += 1
		if non_liver_count >= segmentation_count:
			non_liver_count = 0
			indices.append(i)

	return indices



# delete all the points at the given indices
# points = list of points
# indices = list of indices
def delete_from_indices(points, indices):
	indices.reverse()
	for i in indices:
		points.pop(i)

	return points



# get RGB color values from the float64
# fValue = float64 value from ndarray
def get_rgb_color(fValue):
	# convert float64 to bytearray -> returns 8 Bytes
	# binary[0-4] = 0, binary[5] = r value, binary[6] = g value, binary[7] = b value
	binary = bytes(fValue)
	r = int(binary[5])
	g = int(binary[6])
	b = int(binary[7])
	return (r, g, b)



# find indices of points with specified colors
# colors = ndarray of colors to be filtered
# hsv_threshold = threshold to delete colors
# returns list of indices(int)
def find_color_indices(colors, hsv_threshold):
	indices = []
	for i in range(len(colors)):
		rgb = get_rgb_color(colors[i])
		# rgb_to_hsv() takes inputs between 0 and 1
		# https://docs.python.org/3/library/colorsys.html
		hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)

		if (hsv[2] >= hsv_threshold):
			indices.append(i)

	return indices
