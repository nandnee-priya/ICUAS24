#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float32, Header, Duration, Bool, Int32
import tf
import time
import math
import numpy as np
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint, MultiDOFJointTrajectory
from geometry_msgs.msg import Twist, Transform, PoseStamped, Point, Quaternion
from quadrotor_msgs.msg import PositionCommand
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Imu, Image
from mavros_msgs.srv import SetMode, SetModeRequest
import cv2
from cv_bridge import CvBridge
from collections import deque
bridge = CvBridge()
import tanmay

coord_matrix = [ [0,0,0],
    [4  ,    6, 1.3], [4  ,    6, 4.1], [4  ,    6, 6.9],
    [4  , 13.5, 1.3], [4  , 13.5, 4.1], [4  , 13.5, 6.9],
    [4  ,   21, 1.3], [4  ,   21, 4.1], [4  ,   21, 6.9],
    
    [10 ,    6, 1.3], [10 ,    6, 4.1], [10 ,    6, 6.9],
    [10 , 13.5, 1.3], [10 , 13.5, 4.1], [10 , 13.5, 6.9],
    [10 ,   21, 1.3], [10 ,   21, 4.1], [10 ,   21, 6.9],

    [16 ,    6, 1.3], [16 ,    6, 4.1], [16 ,    6, 6.9],
    [16 , 13.5, 1.3], [16 , 13.5, 4.1], [16 , 13.5, 6.9],
    [16 ,   21, 1.3], [16 ,   21, 4.1], [16 ,   21, 6.9]
]

answer_dictionary = {}

class CONTROLLER:
	def __init__(self):
		self.position = Point()
		self.roll = self.pitch = self.yaw = 0
		self.image = Image()

		rospy.Subscriber("/red/odometry", Odometry , self.update_odometry)
		rospy.Subscriber("/red/camera/color/image_raw", Image, self.update_image)
		rospy.Subscriber("/red/plants_beds", String, self.start_solution)
		
		self.print_image_publisher = rospy.Publisher("/new_images", Image, queue_size = 30)
		self.pose_stamped_publisher = rospy.Publisher("/red/tracker/input_pose", PoseStamped, queue_size = 10)

	def spinfunction(self):
		rospy.spin()

	def update_odometry(self, msg):
		self.position.x = msg.pose.pose.position.x
		self.position.y = msg.pose.pose.position.y
		self.position.z = msg.pose.pose.position.z
		self.roll, self.pitch, self.yaw = self.euler_angles(msg.pose.pose.orientation.x , msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
	
	def update_image(self, msg):
		self.image = msg 

	#converts quaternion to euler
	def euler_angles(self, x, y, z, w):
		#roll, pitch, yaw? or pitch, roll, yaw
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll = math.atan2(t0, t1)
		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch = math.asin(t2)
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw = math.atan2(t3, t4)
		return roll, pitch, yaw

	#converts euler to quaternion
	def quaternions(self, roll, pitch, yaw):
		qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
		qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
		qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
		qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
		return qx, qy, qz, qw

	def start_solution(self, msg):
		parts = msg.data.split()
		name = parts[0]
		bed_list = list(map(int, parts[1:]))

		order_of_traversal = self.generate_order_of_traversal(bed_list)
		#x,y,z,yaw,bed,plant

		for coordinates in order_of_traversal:
			print(coordinates)
			self.go_to_point(coordinates[0],coordinates[1],coordinates[2],coordinates[3])
			Image = self.image
			img = bridge.imgmsg_to_cv2(Image, "bgr8")
			position = (coordinates[4]*3)+coordinates[5]
			if position not in answer_dictionary:
				answer_dictionary[position] = []
			ans_coords = tanmay.get_coords(img, name)
			answer_dictionary[position].append(ans_coords)
			
		total_pairs = 0
		ans = 0
		for key, value in answer_dictionary.items():
			if key == -4:
				continue  # Skip processing key -4
			list1, list2 = value  # Extract the two lists of coordinates
			# Iterate through all possible pairs of points
			total_pairs = 0
			for point1 in list1:
				for point2 in list2:
					# Calculate conditions
					y_diff = abs(point1[1] - point2[1])
					x_sum = point1[0] + point2[0]
					# Check conditions
					if y_diff < 20 and -20 <= x_sum <= 20:
						total_pairs += 1

			ans = ans + len(list1) + len(list2) - total_pairs
			print (len(list1), len(list2), total_pairs)

		fruit_publisher = rospy.Publisher('fruit_count', Int32, queue_size=10)
		print(ans)
		fruit_publisher.publish(ans)	

	def points_inside_polygon(self, coords, square):
		square = np.array(square)
		polygon = cv2.convexHull(square)
		ans = []
		for point in coords:
			if cv2.pointPolygonTest(polygon, tuple(point), False) >= 0:
				ans.append(point)
		return ans

	def position_of_fruit(self, Image, name):
		image = bridge.imgmsg_to_cv2(Image, "bgr8")
		image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
		if name == "Tomato":
			lower = np.array([0, 100, 100])
			upper = np.array([10, 255, 255])
		elif name == "Eggplant":
			lower = np.array([130, 100, 100])
			upper = np.array([160, 255, 255])
		else:
			lower = np.array([20, 100, 100])
			upper = np.array([30, 255, 255])
		mask = cv2.inRange(image_hsv, lower, upper)
		image_final = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
		circle_locations = self.detect_white_shapes(image_final, name)
		Image2 = bridge.cv2_to_imgmsg(image_final, "bgr8")
		self.print_image_publisher.publish(Image2)
		return circle_locations

	def detect_white_shapes(self, image, name):
		hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		lower_white = (0, 0, 200)
		upper_white = (255, 30, 255)
		white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
		white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
		contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		circle_square_locations = []

		# Filter Circles and Squares, find the largest square
		for contour in contours:
			area = cv2.contourArea(contour)
			M = cv2.moments(contour)
			cx = int(M['m10'] / M['m00'])
			cy = int(M['m01'] / M['m00'])		
			print(area, cx, cy)
			# Check for white circles
			if name == "Tomato":
				if area < 1200 and area > 850:
					M = cv2.moments(contour)
					cx = int(M['m10'] / M['m00'])
					cy = int(M['m01'] / M['m00'])
					circle_square_locations.append([cx, cy])
					cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
					cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
			elif name == "Eggplant":
				if area > 3000:
					M = cv2.moments(contour)
					cx = int(M['m10'] / M['m00'])
					cy = int(M['m01'] / M['m00'])
					circle_square_locations.append([cx, cy])
					cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
					cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
			else:
				if area > 3000:
					M = cv2.moments(contour)
					cx = int(M['m10'] / M['m00'])
					cy = int(M['m01'] / M['m00'])
					circle_square_locations.append([cx, cy])
					cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
					cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

		return circle_square_locations

	def find_biggest_square(self, img_gray):
		_, thresh = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)
		contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		largest_contour = None
		max_area = 0
		for contour in contours:
			peri = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
			if len(approx) == 4:
				area = cv2.contourArea(contour)
				if area > max_area:
					largest_contour = approx.reshape(-1, 2)
					max_area = area
		return largest_contour

	def mask_green_and_find_biggest_square(self, image):
		img = bridge.imgmsg_to_cv2(image, "bgr8")
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		lower_green = np.array([40, 40, 40])  # Lower green boundary
		upper_green = np.array([70, 255, 255]) # Upper green boundary
		img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
		mask = cv2.inRange(img_hsv, lower_green, upper_green)
		mask_inverse = cv2.bitwise_not(mask)
		img_rgb[mask_inverse == 0] = [255, 255, 255]
		img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
		square_corners = self.find_biggest_square(img_gray)

		if square_corners is not None:
			return square_corners
		else:
			return [[0,0],[0,0],[0,0],[0,0]]

	def check_if_reached(self, x,y,z):
		rate = rospy.Rate(10)
		dist = 20
		i = 1
		while True:
			if (dist < 1):
				i+=1
				if (i==10):
					break
			dist = math.sqrt((x-self.position.x)**2 + (y-self.position.y)**2 + (z-self.position.z)**2)
			rate.sleep()

	def go_to_point(self, x,y,z,yaw):
		p = PoseStamped()
		p.pose.position.x,p.pose.position.y,p.pose.position.z = x,y,z
		p.pose.orientation.x,p.pose.orientation.y,p.pose.orientation.z,p.pose.orientation.w = self.quaternions(0,0,yaw)
		self.pose_stamped_publisher.publish(p)
		self.check_if_reached(x,y,z)
				
	def give_coordinates(self, bed_number, direction, frontback):
		if direction == "left":
			a = -1.5
			b = 1.5
			p1 = 1
			p2 = 2
			p3 = 3
		else:
			a = 1.5
			b = -1.5
			p1 = 3
			p2 = 2
			p3 = 1

		if frontback == "front":
			h = -2
			yaw = 0
		else:
			h = 2
			yaw = 3.14

		list1 = [coord_matrix[bed_number][0]+h,coord_matrix[bed_number][1]+a,coord_matrix[bed_number][2], yaw, bed_number, p1]
		list2 = [coord_matrix[bed_number][0]+h,coord_matrix[bed_number][1]  ,coord_matrix[bed_number][2], yaw, bed_number, p2]
		list3 = [coord_matrix[bed_number][0]+h,coord_matrix[bed_number][1]+b,coord_matrix[bed_number][2], yaw, bed_number, p3]

		return list1,list2,list3

	def generate_order_of_traversal(self, bed_list):
		run1 = []
		for number in range(1, 10):
			if number in bed_list:
				run1.append(number)
		run2 = []
		for number in range(10, 19):
			if number in bed_list:
				run2.append(number)
		run3 = []
		for number in range(19, 28):
			if number in bed_list:
				run3.append(number)

		coordinate_list = []
		ans = []

		if not run1:
			coordinate_list.append([8, 1, 1.3, 0, -1, -1])
		else:
			if (1 in run1 or 4 in run1 or 7 in run1) or (not(1 in run1 or 4 in run1 or 7 in run1) and not(2 in run1 or 5 in run1 or 8 in run1)):
				templist = [1,4,7,8,5,2,3,6,9]
				for i in range(len(templist)):
					var = templist[i]
					if var in run1:
						dir = "left"
						if var==8 or var==5 or var==2:
							dir = "right"
						list1, list2, list3 = self.give_coordinates(var,dir,"front")
						coordinate_list.append(list1)
						coordinate_list.append(list2)
						coordinate_list.append(list3)
			else:
				templist = [2,5,8,9,6,3]
				for i in range(len(templist)):
					var = templist[i]
					if var in run1:
						dir = "left"
						if var==9 or var==6 or var==3:
							dir = "right"
						list1, list2, list3 = self.give_coordinates(var,dir,"front")
						coordinate_list.append(list1)
						coordinate_list.append(list2)
						coordinate_list.append(list3)

			if (coordinate_list[-1][1] == 4.5 or coordinate_list[-1][1] == 12 or coordinate_list[-1][1] == 19.5 ):
				coordinate_list.append([coordinate_list[-1][0]  ,  1, coordinate_list[-1][2],   0, -1, -1])
				coordinate_list.append([coordinate_list[-1][0]+4,  1, coordinate_list[-1][2],1.56, -1, -1])
			else:
				coordinate_list.append([coordinate_list[-1][0]  , 26, coordinate_list[-1][2],   0, -1, -1])
				coordinate_list.append([coordinate_list[-1][0]+4, 26, coordinate_list[-1][2],1.56, -1, -1])
		
			if (1 in run1 or 4 in run1 or 7 in run1) or (not(1 in run1 or 4 in run1 or 7 in run1) and not(2 in run1 or 5 in run1 or 8 in run1)):
				templist = [9,6,3,2,5,8,7,4,1]
				for i in range(len(templist)):
					var = templist[i]
					if var in run1:
						dir = "right"
						if var==8 or var==5 or var==2:
							dir = "left"
						list1, list2, list3 = self.give_coordinates(var,dir,"back")
						coordinate_list.append(list1)
						coordinate_list.append(list2)
						coordinate_list.append(list3)
			else:
				templist = [3,6,9,8,5,2]
				for i in range(len(templist)):
					var = templist[i]
					if var in run2:
						dir = "right"
						if var==9 or var==6 or var==3:
							dir = "left"
						list1, list2, list3 = self.give_coordinates(var,dir,"back")
						coordinate_list.append(list1)
						coordinate_list.append(list2)
						coordinate_list.append(list3)

		if not run2:
			if (coordinate_list[-1][1] == 4.5 or coordinate_list[-1][1] == 12 or coordinate_list[-1][1] == 19.5 or coordinate_list[-1][1] == 1):
				coordinate_list.append([coordinate_list[-1][0]  ,  1, coordinate_list[-1][2],   0, -1, -1])
			else:
				coordinate_list.append([coordinate_list[-1][0]  , 26, coordinate_list[-1][2],   0, -1, -1])
			coordinate_list.append([coordinate_list[-1][0]+6, coordinate_list[-1][1], coordinate_list[-1][2], coordinate_list[-1][3], -1, -1])
		else:
			if (10 in run2 or 13 in run2 or 16 in run2) or (not(10 in run2 or 13 in run2 or 16 in run2) and not(11 in run2 or 14 in run2 or 17 in run2)):
				templist = [10,13,16,17,14,11,12,15,18]
				for i in range(len(templist)):
					var = templist[i]
					if var in run2:
						dir = "left"
						if var==17 or var==14 or var==11:
							dir = "right"
						list1, list2, list3 = self.give_coordinates(var,dir,"front")
						coordinate_list.append(list1)
						coordinate_list.append(list2)
						coordinate_list.append(list3)
			else:
				templist = [11,14,17,18,15,12]
				for i in range(len(templist)):
					var = templist[i]
					if var in run2:
						dir = "left"
						if var==18 or var==15 or var==12:
							dir = "right"
						list1, list2, list3 = self.give_coordinates(var,dir,"front")
						coordinate_list.append(list1)
						coordinate_list.append(list2)
						coordinate_list.append(list3)

			if (coordinate_list[-1][1] == 4.5 or coordinate_list[-1][1] == 12 or coordinate_list[-1][1] == 19.5 ):
				coordinate_list.append([coordinate_list[-1][0]  ,  1, coordinate_list[-1][2],   0, -1, -1])
				coordinate_list.append([coordinate_list[-1][0]+4,  1, coordinate_list[-1][2],1.56, -1, -1])
			else:
				coordinate_list.append([coordinate_list[-1][0]  , 26, coordinate_list[-1][2],   0, -1, -1])
				coordinate_list.append([coordinate_list[-1][0]+4, 26, coordinate_list[-1][2],1.56, -1, -1])

			if (10 in run2 or 13 in run2 or 16 in run2) or (not(10 in run2 or 13 in run2 or 16 in run2) and not(11 in run2 or 14 in run2 or 17 in run2)):
				templist = [18, 15, 12, 11, 14, 17, 16, 13, 10]
				for i in range(len(templist)):
					var = templist[i]
					if var in run2:
						dir = "right"
						if var==17 or var==14 or var==11:
							dir = "left"
						list1, list2, list3 = self.give_coordinates(var,dir,"back")
						coordinate_list.append(list1)
						coordinate_list.append(list2)
						coordinate_list.append(list3)
			else:
				templist = [12, 15, 18, 17, 14, 11]
				for i in range(len(templist)):
					var = templist[i]
					if var in run2:
						dir = "right"
						if var==18 or var==15 or var==12:
							dir = "left"
						list1, list2, list3 = self.give_coordinates(var,dir,"back")
						coordinate_list.append(list1)
						coordinate_list.append(list2)
						coordinate_list.append(list3)
			
		if run3:
			print(run3)
			print("yes")
			if (19 in run3 or 22 in run3 or 25 in run3) or (not(19 in run3 or 22 in run3 or 25 in run3) and not(20 in run3 or 23 in run3 or 26 in run3)):
				templist = [19,22,25,26,23,20,21,24,27]
				for i in range(len(templist)):
					var = templist[i]
					if var in run3:
						dir = "left"
						if var==17 or var==14 or var==11:
							dir = "right"
						list1, list2, list3 = self.give_coordinates(var,dir,"front")
						coordinate_list.append(list1)
						coordinate_list.append(list2)
						coordinate_list.append(list3)
			else:
				templist = [11,14,17,18,15,12]
				for i in range(len(templist)):
					var = templist[i]
					if var in run3:
						dir = "left"
						if var==18 or var==15 or var==12:
							dir = "right"
						list1, list2, list3 = self.give_coordinates(var,dir,"front")
						coordinate_list.append(list1)
						coordinate_list.append(list2)
						coordinate_list.append(list3)

			if (coordinate_list[-1][1] == 4.5 or coordinate_list[-1][1] == 12 or coordinate_list[-1][1] == 19.5 or coordinate_list[-1][1] == 1):
				coordinate_list.append([coordinate_list[-1][0]  ,  1, coordinate_list[-1][2],   0, -1, -1])
				coordinate_list.append([coordinate_list[-1][0]+4,  1, coordinate_list[-1][2],1.56, -1, -1])
			else:
				coordinate_list.append([coordinate_list[-1][0]  , 26, coordinate_list[-1][2],   0, -1, -1])
				coordinate_list.append([coordinate_list[-1][0]+4, 26, coordinate_list[-1][2],1.56, -1, -1])

			if (19 in run3 or 22 in run3 or 25 in run3) or (not(19 in run3 or 22 in run3 or 25 in run3) and not(20 in run3 or 23 in run3 or 26 in run3)):
				templist = [27, 24, 21, 20, 23, 26, 25, 22, 19]
				for i in range(len(templist)):
					var = templist[i]
					if var in run3:
						dir = "right"
						if var==17 or var==14 or var==11:
							dir = "left"
						list1, list2, list3 = self.give_coordinates(var,dir,"back")
						coordinate_list.append(list1)
						coordinate_list.append(list2)
						coordinate_list.append(list3)
			else:
				templist = [12, 15, 18, 17, 14, 11]
				for i in range(len(templist)):
					var = templist[i]
					if var in run3:
						dir = "right"
						if var==18 or var==15 or var==12:
							dir = "left"
						list1, list2, list3 = self.give_coordinates(var,dir,"back")
						coordinate_list.append(list1)
						coordinate_list.append(list2)
						coordinate_list.append(list3)
		
		return coordinate_list

	def plant_coordinate_creator(self,bed_number, plant_number,yaw):
		if yaw==0:
			x = 4 + ((bed_number-1)//9)*6 - 2
		else:
			x = 4 + ((bed_number-1)//9)*6 + 2
		if plant_number==1:
			y = 6 + (((bed_number-1)//3)%3)*7.5 - 1.5
		elif plant_number==2:
			y = 6 + (((bed_number-1)//3)%3)*7.5 - 0
		else:
			y = 6 + (((bed_number-1)//3)%3)*7.5 + 1.5
		z = 1.3 + ((bed_number-1)%3)*2.8

		return x,y,z

		
def mainCall():
	rospy.init_node('Trajectory_Publisher_Node', anonymous = True)
	drone = CONTROLLER()
	print("Program started")
	drone.spinfunction()

if __name__ == '__main__':
	try:
		mainCall()
	except rospy.ROSInterruptException:
		print("ROS Terminated")
		pass
