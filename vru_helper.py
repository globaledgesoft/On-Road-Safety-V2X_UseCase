import global_var 
from helper.utils.detection_helper import get_centroids_and_groundpoints_car
import cv2
import numpy as np
from collections import deque
# import rospy
from helper.utils.constants import *

def check_point_in_zone(x, y, frame_test, r_points_static):
	x,y = int(x), int(y)
	flag = FLAG_FALSE
	if (x >= 0 or x < frame_test.shape[1]) and (y >= 0 or y < frame_test.shape[0]):
		dist1 = cv2.pointPolygonTest(r_points_static.astype(int), (x, y), True)
		if dist1 >= ZONE_DIST_THRESH:
			flag = FLAG_TRUE
	else:
		if x < 0:
			x = 0
		if y < 0:
			y = 0
		if x >= frame_test.shape[1]:
			x = frame_test.shape[1] - 1 
		if y >= frame_test.shape[0]:
			y = frame_test.shape[0] - 1 
		flag = FLAG_FALSE
	return x, y, flag

def draw_circle_wrt_flag(frame_test, frame_test_vis, direction_img, x, y, flag):
	if flag == FLAG_TRUE :
		cv2.circle(frame_test, (x,y), RADIOUS_3, COLOR_RED, FILL_SHAPE)
		cv2.circle(frame_test_vis, (x,y), RADIOUS_3, COLOR_RED, FILL_SHAPE)
		cv2.circle(direction_img, (x,y), RADIOUS_3, COLOR_RED, -1)
		
	elif flag == FLAG_FALSE:
		cv2.circle(frame_test, (x,y), RADIOUS_3, COLOR_GREEN, FILL_SHAPE)
		cv2.circle(frame_test_vis, (x,y), RADIOUS_3, COLOR_GREEN, FILL_SHAPE)

def update_global_car_tracking_list(tracks, f_cnt, frame_test, frame_test_vis, frame_test_inter, direction_img, r_points_static):

	frame_test_car_in_roi_boxes = []
	
	for track in tracks:
		# for car
		array_centroids_car, array_groundpoints_car = get_centroids_and_groundpoints_car(track.box)
		
		for point_i in range(0, len(array_centroids_car), 6):
			top_left = array_centroids_car[point_i]
			top_right = array_centroids_car[point_i+1]

			bott_left = array_centroids_car[point_i+2]
			bot_right = array_centroids_car[point_i+3]

			centroid = array_centroids_car[point_i+4]
			ground_point = array_centroids_car[point_i+5]
			
			x,y = bot_right
			x, y, flag = check_point_in_zone(x, y, frame_test, r_points_static)
			draw_circle_wrt_flag(frame_test, frame_test_vis, direction_img, x, y, flag)

			x,y = bott_left
			x, y, flag = check_point_in_zone(x, y, frame_test, r_points_static)
			draw_circle_wrt_flag(frame_test, frame_test_vis, direction_img, x, y, flag)

			x,y = ground_point
			x, y, flag = check_point_in_zone(x, y, frame_test, r_points_static)
			draw_circle_wrt_flag(frame_test, frame_test_vis, direction_img, x, y, flag)

			# get currunt box
			x11, y11, x22, y22 = int(track.box[0]), int(track.box[1]), int(track.box[2]), int(track.box[3])
		
			if flag == FLAG_TRUE:
				cv2.rectangle(frame_test_inter, (x11, y11), (x22, y22), COLOR_PINK, THICKNESS_2)
				frame_test_car_in_roi_boxes.append([x11, y11, x22, y22, track.id])
			else:
				cv2.rectangle(frame_test_inter, (x11, y11), (x22, y22), COLOR_BLUE, THICKNESS_2)

			# # track centroied
			# x,y = centroid
			# center_car = (x,y)

			# track ground point
			x,y = ground_point
			center_car = (x,y)

			if f_cnt == 1:
				global_var.car_moving_points = deque(maxlen=QUEUE_LEN_32)
				global_var.car_moving_rect = deque(maxlen=QUEUE_LEN_32)
				#0 car_moving_points
				global_var.global_car_track_dict[track.id] = list()
				global_var.global_car_track_dict[track.id].append(global_var.car_moving_points)
				global_var.global_car_track_dict[track.id][0].appendleft(center_car)
				#1 car_moving_rect
				global_var.global_car_track_dict[track.id].append(global_var.car_moving_rect)
				global_var.global_car_track_dict[track.id][1].appendleft([x11, y11, x22, y22])
				#2 car_count
				global_var.global_car_track_dict[track.id].append([1])
				#3 move_car_votes
				move_car_votes = deque(maxlen=5)
				global_var.global_car_track_dict[track.id].append(move_car_votes)
				#4 unique_id_car
				global_var.unique_id_car += 1
				global_var.global_car_track_dict[track.id].append([global_var.unique_id_car])
				#5 in zone flag
				if flag == 1:
					in_zone_flag = True
					global_var.global_car_track_dict[track.id].append([in_zone_flag])
				else:
					in_zone_flag = False
					global_var.global_car_track_dict[track.id].append([in_zone_flag])
				#6 risk_level_flags -1 0 1 2
				global_var.global_car_track_dict[track.id].append([0])
				#7 rospy_time
				# global_var.global_car_track_dict[track.id].append([str(rospy.get_time())])
				global_var.global_car_track_dict[track.id].append([str("time")])
				#8 line of sight
				los = None
				global_var.global_car_track_dict[track.id].append([los])
				#9 currunt ground points list
				global_var.global_car_track_dict[track.id].append([bott_left, bot_right, ground_point])

			else:
				if track.id in global_var.global_car_track_dict:
					#0 car_moving_points
					global_var.global_car_track_dict[track.id][MOV_POINTS_IND].appendleft(center_car)
					#1 car_moving_rect
					global_var.global_car_track_dict[track.id][MOV_RECT_IND].appendleft([x11, y11, x22, y22])
					#2 car_count
					global_var.global_car_track_dict[track.id][OBJ_CNT_IND][0] = global_var.global_car_track_dict[track.id][OBJ_CNT_IND][0] + 1
					#3 move_car_votes
					#4 unique_id_car
					#5 in zone flag
					if flag == FLAG_TRUE:
						in_zone_flag = True
						global_var.global_car_track_dict[track.id][IN_ZONE_IND][0] = in_zone_flag
					else:
						in_zone_flag = False
						global_var.global_car_track_dict[track.id][IN_ZONE_IND][0] = in_zone_flag
					#6 risk level
					global_var.global_car_track_dict[track.id][EVENT_TYPE_IND][0] = 0
					#7 rospy_time
					# global_var.global_car_track_dict[track.id][TIME_IND][0] = str(rospy.get_time())
					global_var.global_car_track_dict[track.id][TIME_IND][0] = str("time")
					#9 currunt ground points list
					global_var.global_car_track_dict[track.id][G_POINTS_IND] = [bott_left, bot_right, ground_point]
	
				else:
					global_var.car_moving_points = deque(maxlen=QUEUE_LEN_32)
					global_var.car_moving_rect = deque(maxlen=QUEUE_LEN_32)
					
					global_var.global_car_track_dict[track.id] = list()
					global_var.global_car_track_dict[track.id].append(global_var.car_moving_points)
					global_var.global_car_track_dict[track.id][0].appendleft(center_car)

					global_var.global_car_track_dict[track.id].append(global_var.car_moving_rect)
					global_var.global_car_track_dict[track.id][1].appendleft([x11, y11, x22, y22])
					global_var.global_car_track_dict[track.id].append([1])
					move_car_votes = deque(maxlen=5)
					global_var.global_car_track_dict[track.id].append(move_car_votes)

					global_var.unique_id_car += 1
					global_var.global_car_track_dict[track.id].append([global_var.unique_id_car])
					#5 in zone flag
					if flag == FLAG_TRUE:
						in_zone_flag = True
						global_var.global_car_track_dict[track.id].append([in_zone_flag])
					else:
						in_zone_flag = False
						global_var.global_car_track_dict[track.id].append([in_zone_flag])
					#6 risk_level_flags -1 0 1 2
					global_var.global_car_track_dict[track.id].append([0])
					#7 rospy_time
					# global_var.global_car_track_dict[track.id].append([str(rospy.get_time())])
					global_var.global_car_track_dict[track.id].append([str("time")])
					#8 line of sight
					los = None
					global_var.global_car_track_dict[track.id].append([los])
					#9 currunt ground points list
					global_var.global_car_track_dict[track.id].append([bott_left, bot_right, ground_point])
	return frame_test_car_in_roi_boxes

def update_global_ped_tracking_list(tracks_ped, f_cnt, frame_test, frame_test_vis, frame_test_inter, direction_img, r_points_static):

	frame_test_ped_in_roi_boxes = []
	for track in tracks_ped:
		array_centroids_ped, array_groundpoints_ped = get_centroids_and_groundpoints_car(track.box)
		
		for point_i in range(0, len(array_centroids_ped), 6):
			top_left = array_centroids_ped[point_i]
			top_right = array_centroids_ped[point_i+1]

			bott_left = array_centroids_ped[point_i+2]
			bot_right = array_centroids_ped[point_i+3]

			centroid = array_centroids_ped[point_i+4]
			ground_point = array_centroids_ped[point_i+5]
			
			x,y = bot_right
			x, y, flag = check_point_in_zone(x, y, frame_test, r_points_static)
			draw_circle_wrt_flag(frame_test, frame_test_vis, direction_img, x, y, flag)

			x,y = bott_left
			x, y, flag = check_point_in_zone(x, y, frame_test, r_points_static)
			draw_circle_wrt_flag(frame_test, frame_test_vis, direction_img, x, y, flag)

			x,y = ground_point
			x, y, flag = check_point_in_zone(x, y, frame_test, r_points_static)
			draw_circle_wrt_flag(frame_test, frame_test_vis, direction_img, x, y, flag)

			# get curunt  tracked box
			x11, y11, x22, y22 = int(track.box[0]), int(track.box[1]), int(track.box[2]), int(track.box[3])
		
			if flag == FLAG_TRUE:
				cv2.rectangle(frame_test_inter, (x11, y11), (x22, y22), COLOR_PINK, THICKNESS_2)
				frame_test_ped_in_roi_boxes.append([x11, y11, x22, y22, track.id])

			else:
				cv2.rectangle(frame_test_inter, (x11, y11), (x22, y22), COLOR_BLUE, THICKNESS_2)

			# # track centroied
			# x,y = centroid
			# center_ped = (x,y)

			# track ground_point
			x,y = ground_point
			center_ped = (x,y)

			if f_cnt == 1:
				global_var.ped_moving_points = deque(maxlen=QUEUE_LEN_32)
				global_var.ped_moving_rect = deque(maxlen=QUEUE_LEN_32)
				
				global_var.global_ped_track_dict[track.id] = list()
				global_var.global_ped_track_dict[track.id].append(global_var.ped_moving_points)
				global_var.global_ped_track_dict[track.id][0].appendleft(center_ped)

				global_var.global_ped_track_dict[track.id].append(global_var.ped_moving_rect)
				global_var.global_ped_track_dict[track.id][1].appendleft([x11, y11, x22, y22])
				global_var.global_ped_track_dict[track.id].append([1])

				move_ped_votes = deque(maxlen=5)
				global_var.global_ped_track_dict[track.id].append(move_ped_votes)

				global_var.unique_id_ped += 1
				global_var.global_ped_track_dict[track.id].append([global_var.unique_id_ped])
				#5 in zone flag
				if flag == FLAG_TRUE:
					in_zone_flag = True
					global_var.global_ped_track_dict[track.id].append([in_zone_flag])
				else:
					in_zone_flag = False
					global_var.global_ped_track_dict[track.id].append([in_zone_flag])
				#6 risk_level_flags -1 0 1 2
				global_var.global_ped_track_dict[track.id].append([0])
				#7 rospy_time
				# global_var.global_ped_track_dict[track.id].append([str(rospy.get_time())])
				global_var.global_ped_track_dict[track.id].append([str("time")])
				#8 line of sight
				los = None
				global_var.global_ped_track_dict[track.id].append([los])
				global_var.global_ped_track_dict[track.id][LOS_IND].append(los)
				global_var.global_ped_track_dict[track.id][LOS_IND].append(los)
				#9 currunt ground points list
				global_var.global_ped_track_dict[track.id].append([bott_left, bot_right, ground_point])

			else:
				if track.id in global_var.global_ped_track_dict:
					global_var.global_ped_track_dict[track.id][MOV_POINTS_IND].appendleft(center_ped)
					global_var.global_ped_track_dict[track.id][MOV_RECT_IND].appendleft([x11, y11, x22, y22])
					global_var.global_ped_track_dict[track.id][OBJ_CNT_IND][0] = global_var.global_ped_track_dict[track.id][OBJ_CNT_IND][0] + 1
					#5 in zone flag
					if flag == FLAG_TRUE:
						in_zone_flag = True
						global_var.global_ped_track_dict[track.id][IN_ZONE_IND][0] = in_zone_flag
					else:
						in_zone_flag = False
						global_var.global_ped_track_dict[track.id][IN_ZONE_IND][0] = in_zone_flag
					#6 risk_level_flags -1 0 1 2
					global_var.global_ped_track_dict[track.id][EVENT_TYPE_IND][0] = 0
					#7 rospy_time
					# global_var.global_ped_track_dict[track.id][TIME_IND][0] = str(rospy.get_time())
					global_var.global_ped_track_dict[track.id][TIME_IND][0] = str("time")
					#9 currunt ground points list
					global_var.global_ped_track_dict[track.id][G_POINTS_IND] = [bott_left, bot_right, ground_point]
				else:
					global_var.ped_moving_points = deque(maxlen=QUEUE_LEN_32)
					global_var.ped_moving_rect = deque(maxlen=QUEUE_LEN_32)
					
					global_var.global_ped_track_dict[track.id] = list()
					global_var.global_ped_track_dict[track.id].append(global_var.ped_moving_points)
					global_var.global_ped_track_dict[track.id][0].appendleft(center_ped)

					global_var.global_ped_track_dict[track.id].append(global_var.ped_moving_rect)
					global_var.global_ped_track_dict[track.id][1].appendleft([x11, y11, x22, y22])
					global_var.global_ped_track_dict[track.id].append([1])

					move_ped_votes = deque(maxlen=5)
					global_var.global_ped_track_dict[track.id].append(move_ped_votes)

					global_var.unique_id_ped += 1
					global_var.global_ped_track_dict[track.id].append([global_var.unique_id_ped])
					#5 in zone flag
					if flag == FLAG_TRUE:
						in_zone_flag = True
						global_var.global_ped_track_dict[track.id].append([in_zone_flag])
					else:
						in_zone_flag = False
						global_var.global_ped_track_dict[track.id].append([in_zone_flag])
					#6 risk_level_flags -1 0 1 2
					global_var.global_ped_track_dict[track.id].append([0])
					#7 rospy_time
					# global_var.global_ped_track_dict[track.id].append([str(rospy.get_time())])
					global_var.global_ped_track_dict[track.id].append([str("time")])
					#8 line of sight
					los = None
					global_var.global_ped_track_dict[track.id].append([los])
					global_var.global_ped_track_dict[track.id][LOS_IND].append(los)
					global_var.global_ped_track_dict[track.id][LOS_IND].append(los)
					#9 currunt ground points list
					global_var.global_ped_track_dict[track.id].append([bott_left, bot_right, ground_point])

	return frame_test_ped_in_roi_boxes