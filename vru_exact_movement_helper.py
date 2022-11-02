import global_var 
import cv2
import numpy as np
from helper.utils.constants import *

def draw_directions(direction_img, x11, y11, x22, y22, angle, color):
		length = x22-x11
		# # for centroied
		# x1, y1 = (x11 + ((x22-x11)/2), y11 + ((y22-y11)/2))

		# for ground point
		x1, y1 = x11 + ((x22-x11)/2), y22
		x1, y1 = int(x1), int(y1)
		x2 =  int(round(x1 + length * np.cos(angle * PI / DEG_180)))
		y2 =  int(round(y1 + length * np.sin(angle * PI / DEG_180)))
		cv2.line(direction_img, (x1,y1), (x2,y2), color, THICKNESS_2)
		line = [(x1,y1), (x2,y2)]
		return line

def get_moving_vehicle_VA(c_sight, line, direction_img):
	x_inter, y_inter = line_intersection(c_sight, line)
	(x0,y0), (x1,y1) = c_sight[0],c_sight[1]

	if x_inter != None or y_inter != None:
		if ((x1 <= x0 and x_inter <= x0) or (x1 >= x0 and x_inter >= x0)) and ((y1 <= y0 and y_inter <= y0) or (y1 >= y0 and y_inter >= y0)):
			if x_inter < 0 or x_inter > direction_img.shape[1] or y_inter < 0 or y_inter > direction_img.shape[0]:
				return None, None
			else:
				x_inter, y_inter = int(x_inter), int(y_inter)
				return [np.abs(x_inter), np.abs(y_inter)], line	
	return None, None

def line_intersection(line1, line2):
	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	x,y = None, None
	if div == 0:
		return x, y

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return x, y

def find_angel_bet_2_lines(direction_img, x11, y11, x22, y22, disp_x, disp_y):

	# # for centroied tracker
	# x1,y1 = x22, y11 + ((y22-y11)/2) #p0
	# x1,y1 = int(x1), int(y1)
	# p0 = [x1,y1]

	# x2,y2 = (x11 + (x22-x11)/2), y11 + ((y22-y11)/2) #p1
	# x2,y2 = int(x2), int(y2)
	# p1 = [x2,y2]

	# x3, y3 = x11 + ((x22-x11)/2) - disp_x, (y11 + ((y22-y11)/2)) - disp_y #p2
	# x3, y3 = int(x3), int(y3)
	# p2 = [x3, y3]

	# for ground point tracker
	# add code here
	x1,y1 = x22, y22#p0
	x1,y1 = int(x1), int(y1)
	p0 = [x1,y1]

	x2,y2 = (x11 + (x22-x11)/2), y22 #p1
	x2,y2 = int(x2), int(y2)
	p1 = [x2,y2]

	x3, y3 = (x11 + (x22-x11)/2)  - disp_x, y22 - disp_y #p2
	x3, y3 = int(x3), int(y3)
	p2 = [x3, y3]

	v0 = np.array(p0) - np.array(p1)
	v1 = np.array(p2) - np.array(p1)

	theta = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
	angel = np.degrees(theta)
	draw_directions(direction_img, x11, y11, x22, y22, angel, COLOR_GREEN)
	side_angel1 = angel - OBJ_VIEW_ANGLE
	if side_angel1 <= -(DEG_180):
		diff = side_angel1 - (-DEG_180)
		side_angel1 = DEG_180 + diff

	side_angel2 = angel + OBJ_VIEW_ANGLE
	if side_angel2 > DEG_180:
		diff = side_angel2 - DEG_180
		side_angel2 = -DEG_180 + diff

	sight1_line = draw_directions(direction_img, x11, y11, x22, y22, side_angel1, COLOR_BLUE)
	sight2_line = draw_directions(direction_img, x11, y11, x22, y22, side_angel2, COLOR_BLUE)
	los = (x2, y2), (x3, y3)

	return los, sight1_line, sight2_line

def update_global_car_tracking_with_movement(frame_test_vis, frame_test, frame_test_inter, frame_test_vis_table, direction_img, frame_test_car_in_roi_boxes, frame_test_ped_in_roi_boxes, row_cnt):
	# for table visulaization
	x_off = frame_test.shape[0] / 6
	y_off = frame_test.shape[0] / 10

	for id in global_var.global_car_track_dict:
		track_id_data = global_var.global_car_track_dict[id]
		car_moving_p = track_id_data[MOV_POINTS_IND]
		car_moving_r = track_id_data[MOV_RECT_IND]
		id_counter_car = track_id_data[OBJ_CNT_IND][0]

		dX_car = 0
		dY_car = 0
		direction_car = ""
		(dirX, dirY) = ("", "")
		disp_x = 0
		disp_y = 0

		for i in np.arange(1, len(car_moving_p)):
			# if either of the tracked points are None, ignore
			# them
			if car_moving_p[i - 1] is None or car_moving_p[i] is None:
				continue
			# check to see if enough points have been accumulated in
			# the buffer

			# check object moving or not
			if id_counter_car >= N_FRAMES_FOR_MOV_OBJ and i == 1 and car_moving_p[-N_FRAMES_FOR_MOV_OBJ] is not None:
				is_move_x = car_moving_p[-N_FRAMES_FOR_MOV_OBJ][0] - car_moving_p[i][0]
				is_move_y = car_moving_p[-N_FRAMES_FOR_MOV_OBJ][1] - car_moving_p[i][1]

				if (np.abs(is_move_x) > 0) or (np.abs(is_move_y) > 0):
					global_var.global_car_track_dict[id][MOV_VOTE_IND].appendleft(1)
				else:
					global_var.global_car_track_dict[id][MOV_VOTE_IND].appendleft(0)

			if id_counter_car >= N_FRAMES_FOR_DIR_EST and i == 1 and car_moving_p[-N_FRAMES_FOR_DIR_EST] is not None:
				move_x = car_moving_p[-N_FRAMES_FOR_DIR_EST][0] - car_moving_p[i][0]
				move_y = car_moving_p[-N_FRAMES_FOR_DIR_EST][1] - car_moving_p[i][1]
				
				if np.abs(move_x) > 1:
					disp_x = move_x

				if np.abs(move_y) > 1:
					disp_y = move_y
			
			if id_counter_car >= N_FRAMES_FOR_DIR_EST and i == 1 and car_moving_p[-N_FRAMES_FOR_DIR_EST] is not None:
				# compute the difference between the x and y
				# coordinates and re-initialize the direction
				# text variables
				dX_car = car_moving_p[-N_FRAMES_FOR_DIR_EST][0] - car_moving_p[i][0]
				dY_car = car_moving_p[-N_FRAMES_FOR_DIR_EST][1] - car_moving_p[i][1]

				(dirX, dirY) = ("", "")

				# ensure there is significant movement in the
				# x-direction
				if np.abs(dX_car) >= 2:
					dirX = "East" if np.sign(dX_car) == 1 else "West"
				# ensure there is significant movement in the
				# y-direction
				if np.abs(dY_car) >= 2:
					dirY = "North" if np.sign(dY_car) == 1 else "South"
				# handle when both directions are non-empty
				if dirX != "" and dirY != "":
					direction_car = "{}-{}".format(dirY[:2], dirX)
				# otherwise, only one direction is non-empty
				else:
					direction_car = dirX if dirX != "" else dirY

			thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
			color = global_var.unique_colors[global_var.global_car_track_dict[id][UNI_ID_IND][0]%100]
			cv2.line(frame_test_vis, car_moving_p[i - 1], car_moving_p[i], color, thickness)
			cv2.line(frame_test, car_moving_p[i - 1], car_moving_p[i], color, thickness)
			cv2.line(frame_test_inter, car_moving_p[i - 1], car_moving_p[i], color, thickness)

		color = global_var.unique_colors[global_var.global_car_track_dict[id][UNI_ID_IND][0]%100]
		if id_counter_car < N_FRAMES_FOR_MOV_OBJ:
			cv2.putText(frame_test_vis_table, "Car" , (int(0*x_off)+5,int (row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

			cv2.putText(frame_test_vis_table, direction_car , (int(2*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

		cv2.putText(frame_test_vis_table, "{},{}".format(dX_car, dY_car) , (int(3*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

		cv2.circle(frame_test_vis_table, (int(1*x_off + (x_off/2))-5, int(row_cnt*y_off+ (y_off/2)-5)), RADIOUS_5, color, FILL_SHAPE)

		x11, y11, x22, y22 = car_moving_r[0]
		cv2.rectangle(direction_img, (int(x11), int(y11)), (int(x22), int(y22)), color, THICKNESS_2)

		if np.abs(disp_x) > 1 or np.abs(disp_y) > 1:
			los, sight1_line, sight2_line = find_angel_bet_2_lines(direction_img, x11, y11, x22, y22, disp_x, disp_y)
		
		if id_counter_car >= N_FRAMES_FOR_MOV_OBJ:
			move_vote = 0
			for car_i in global_var.global_car_track_dict[id][3]:
				if car_i == 1:
					move_vote += 1

			if move_vote > MOVING_VOTE_THRESH:
				x11, y11, x22, y22 = car_moving_r[0]
				mask1 = cv2.rectangle((frame_test_inter[:,:,1]).astype("uint8"), (x11, y11), (x22, y22), (75), FILL_SHAPE)
				frame_test_inter[:,:,1] = (frame_test_inter[:,:,1]).astype("uint8") | mask1

				cv2.putText(frame_test_vis_table, "Car" , (int(0*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, direction_car , (int(2*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, "moving" , (int(4*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)
			else:
				cv2.putText(frame_test_vis_table, "Car" , (int(0*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, direction_car , (int(2*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, "not moving" , (int(4*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

		if len(frame_test_car_in_roi_boxes) > 0 and len(frame_test_ped_in_roi_boxes) > 0:
			for box in frame_test_car_in_roi_boxes:
				x11, y11, x22, y22, box_id = box
				if box_id == id:
					# setting warning risk event for car
					global_var.global_car_track_dict[id][EVENT_TYPE_IND][0] = 1
					cv2.rectangle(frame_test_vis_table, (int(5*x_off)+3, int(row_cnt*y_off)+3), (int(5*x_off+x_off)-3, int(row_cnt*y_off+y_off)-3), COLOR_PINK, FILL_SHAPE)

					for ped_box in frame_test_ped_in_roi_boxes:
						x11, y11, x22, y22, ped_box_id = ped_box
						# getting line of sight from ped dict
						if global_var.global_ped_track_dict[ped_box_id][8][0] != None and (np.abs(disp_x) > 1 or np.abs(disp_y) > 1):
							p_los = global_var.global_ped_track_dict[ped_box_id][LOS_IND][0]
							p_sight1 = global_var.global_ped_track_dict[ped_box_id][LOS_IND][1]
							p_sight1 = global_var.global_ped_track_dict[ped_box_id][LOS_IND][2]

							c_los = los
							c_sight1 = sight1_line
							c_sight2 = sight2_line

						    # line of sight collision
							x_col, y_col = line_intersection(c_los, p_los)
							if x_col != None and y_col != None:
								x_col, y_col = int(x_col), int(y_col)
								(x0,y0), (x1,y1) = c_los
								(x2,y2), (x3,y3) = p_los

								w_extend = int((x22 - x11)/2)

								def extend(val, x11, y11, x22, y22):
									lenAB = np.sqrt(np.square(x11 - x22) + np.square(y11 - y22))
									x1 = x11 + (x11 - x22) / lenAB * val
									y1 = y11 + (y11 - y22) / lenAB * val
									return int(x1),int(y1)

								ex_x2, ex_y2 = extend(w_extend, x2,y2, x3,y3)
								cv2.line(direction_img, (ex_x2, ex_y2), (x2,y2), (0,255,255), 2)
								(x2,y2) = (ex_x2, ex_y2)
								
								# check collision
								if ((x3 <= x2 and x_col <= x2) or (x3 >= x2 and x_col >=x2)) and \
								((y3 <= y2 and y_col <= y2) or (y3 >= y2 and y_col >= y2)) and \
								((x1 <= x0 and x_col <= x0) or (x1 >= x0 and x_col >= x0)) and \
								((y1 <= y0 and y_col <= y0) or (y1 >= y0 and y_col >= y0)):

									# collision out of frame
									if x_col < 0 or x_col >= direction_img.shape[1] or y_col < 0 or y_col >= direction_img.shape[0]:
										global_var.global_car_track_dict[id][EVENT_TYPE_IND][0] = 1
									
									else:
										global_var.global_car_track_dict[id][EVENT_TYPE_IND][0] = 2
										cv2.circle(direction_img, (x_col, y_col), RADIOUS_3, COLOR_RED, FILL_SHAPE)
										cv2.circle(direction_img, (x_col, y_col), RADIOUS_9, COLOR_RED, THICKNESS_2)

										cv2.line(direction_img, (x1,y1), (x_col, y_col), COLOR_RED, THICKNESS_2)
										cv2.line(direction_img, (x3,y3), (x_col, y_col), COLOR_RED, THICKNESS_2)

										cv2.rectangle(frame_test_vis_table, (int(5*x_off)+3, int(row_cnt*y_off)+3), (int(5*x_off+x_off)-3, int(row_cnt*y_off+y_off)-3), COLOR_RED, FILL_SHAPE)
										break

							
						if (global_var.global_car_track_dict[id][EVENT_TYPE_IND][0] == 1 or global_var.global_car_track_dict[id][EVENT_TYPE_IND][0] == 0 or global_var.global_car_track_dict[id][EVENT_TYPE_IND][0] == -1) and (np.abs(disp_x) > 1 or np.abs(disp_y) > 1):

							c_los = los
							c_sight1 = sight1_line
							c_sight2 = sight2_line

							up_line = [(direction_img.shape[1]-1,0), (0,0)]
							left_line = [(0,0), (0,direction_img.shape[0]-1)]
							bott_line = [(0,direction_img.shape[0]-1), (direction_img.shape[1]-1, direction_img.shape[0]-1)]
							right_line = [(direction_img.shape[1]-1, direction_img.shape[0]-1), (direction_img.shape[1]-1, 0)]

							vehicle_VA_points = []
							vehicle_VA_lines = []
							vehicle_VA = []
							p1, line = get_moving_vehicle_VA(c_sight1, up_line, direction_img)
							if p1 != None and line != None:
								vehicle_VA_points.append(p1)
								vehicle_VA_lines.append(line)

							if p1 == None and line == None:
								p1, line = get_moving_vehicle_VA(c_sight1, left_line, direction_img)
								if p1 != None and line != None:
									vehicle_VA_points.append(p1)
									vehicle_VA_lines.append(line)

							if p1 == None and line == None:
								p1, line = get_moving_vehicle_VA(c_sight1, bott_line, direction_img)
								if p1 != None and line != None:
									vehicle_VA_points.append(p1)
									vehicle_VA_lines.append(line)
									
							if p1 == None and line == None:
								p1, line = get_moving_vehicle_VA(c_sight1, right_line, direction_img)
								if p1 != None and line != None:
									vehicle_VA_points.append(p1)
									vehicle_VA_lines.append(line)

							p1, line = get_moving_vehicle_VA(c_sight2, up_line, direction_img)
							if p1 != None and line != None:
								vehicle_VA_points.append(p1)
								vehicle_VA_lines.append(line)

							if p1 == None and line == None:
								p1, line = get_moving_vehicle_VA(c_sight2, left_line, direction_img)
								if p1 != None and line != None:
									vehicle_VA_points.append(p1)
									vehicle_VA_lines.append(line)

							if p1 == None and line == None:
								p1, line = get_moving_vehicle_VA(c_sight2, bott_line, direction_img)
								if p1 != None and line != None:
									vehicle_VA_points.append(p1)
									vehicle_VA_lines.append(line)

							if p1 == None and line == None:
								p1, line = get_moving_vehicle_VA(c_sight2, right_line, direction_img)
								if p1 != None and line != None:
									vehicle_VA_points.append(p1)
									vehicle_VA_lines.append(line)

							cx1,cy1  = c_sight1[0]
							x_in1, y_in1 = vehicle_VA_points[0]
							(x1,y1), (x2,y2) = vehicle_VA_lines[0]

							cx2,cy2  = c_sight2[0]
							x_in2, y_in2 = vehicle_VA_points[1]
							(x3,y3), (x4,y4) = vehicle_VA_lines[1]

							#9 currunt ground points list
							bott_left = global_var.global_ped_track_dict[ped_box_id][G_POINTS_IND][0]
							bott_right = global_var.global_ped_track_dict[ped_box_id][G_POINTS_IND][1]
							ground_point = global_var.global_ped_track_dict[ped_box_id][G_POINTS_IND][2]
							flag = FLAG_FALSE

							if x1 == x3 and y1 == y3 and x2 == x4 and y2 == y4:
								vehicle_VA = [(cx1, cy1), (x_in1, y_in1), (x_in2, y_in2)]
								vehicle_VA = np.array(vehicle_VA)
								cv2.drawContours(direction_img,[vehicle_VA],CONTOUR_IDX,COLOR_RED,THICKNESS_2)

							elif (x1 == x3 and y1 == y3) or (x1 == x4 and y1 == y4):
								vehicle_VA = [(x_in1, y_in1), (cx1, cy1), (x_in2, y_in2), (x1,y1)]
								vehicle_VA = np.array(vehicle_VA)
								cv2.drawContours(direction_img,[vehicle_VA],CONTOUR_IDX,COLOR_RED,THICKNESS_2)
								
							elif (x2 == x3 and y2 == y3) or (x2 == x4 and y2 == y4):
								vehicle_VA = [(x_in1, y_in1), (cx1, cy1), (x_in2, y_in2), (x2,y2)]
								vehicle_VA = np.array(vehicle_VA)
								cv2.drawContours(direction_img,[vehicle_VA],CONTOUR_IDX,COLOR_RED,THICKNESS_2)

							#right left down
							elif ((cx1 > x_in1) and (cy1 < y_in1)) and ((cx1 < x_in2) and (cy1 < y_in2)):
								vehicle_VA = [(x_in1, y_in1), (cx1, cy1), (x_in2, y_in2), right_line[0], left_line[1]]
								vehicle_VA = np.array(vehicle_VA)
								cv2.drawContours(direction_img,[vehicle_VA],0,(0,0,255),2)
								
							elif ((cx1 > x_in2) and (cy1 < y_in2)) and ((cx1 < x_in1) and (cy1 < y_in1)):
								vehicle_VA = [(x_in1, y_in1), (cx1, cy1), (x_in2, y_in2), left_line[1], right_line[0]]
								vehicle_VA = np.array(vehicle_VA)
								cv2.drawContours(direction_img,[vehicle_VA],0,(0,0,255),2)
			

							#right left up
							elif ((cx1 > x_in1) and (cy1 > y_in1)) and ((cx1 < x_in2) and (cy1 > y_in2)):
								vehicle_VA = [(x_in1, y_in1), (cx1, cy1), (x_in2, y_in2), right_line[1], left_line[0]]
								vehicle_VA = np.array(vehicle_VA)
								cv2.drawContours(direction_img,[vehicle_VA],0,(0,0,255),2)
								
							elif ((cx1 > x_in2) and (cy1 > y_in2)) and ((cx1 < x_in1) and (cy1 > y_in1)):
								vehicle_VA = [(x_in1, y_in1), (cx1, cy1), (x_in2, y_in2), left_line[0], right_line[1]]
								vehicle_VA = np.array(vehicle_VA)
								cv2.drawContours(direction_img,[vehicle_VA],0,(0,0,255),2)

							#up down right
							elif ((cx1 < x_in2) and (cy1 > y_in2)) and ((cx1 < x_in1) and (cy1 < y_in1)):
								vehicle_VA = [(x_in1, y_in1), (cx1, cy1), (x_in2, y_in2), up_line[0], bott_line[1]]
								vehicle_VA = np.array(vehicle_VA)
								cv2.drawContours(direction_img,[vehicle_VA],0,(0,0,255),2)

							elif ((cx1 < x_in1) and (cy1 > y_in1)) and ((cx1 < x_in2) and (cy1 < y_in2)):
								vehicle_VA = [(x_in1, y_in1), (cx1, cy1), (x_in2, y_in2), bott_line[1], up_line[0]]
								vehicle_VA = np.array(vehicle_VA)
								cv2.drawContours(direction_img,[vehicle_VA],0,(0,0,255),2)

							#up down left
							elif ((cx1 > x_in1) and (cy1 > y_in1)) and ((cx1 > x_in2) and (cy1 < y_in2)):
								vehicle_VA = [(x_in1, y_in1), (cx1, cy1), (x_in2, y_in2), bott_line[0], up_line[1]] 
								vehicle_VA = np.array(vehicle_VA)
								cv2.drawContours(direction_img,[vehicle_VA],0,(0,0,255),2)

							elif ((cx1 > x_in2) and (cy1 > y_in2)) and ((cx1 > x_in1) and (cy1 < y_in1)):
								vehicle_VA = [(x_in1, y_in1), (cx1, cy1), (x_in2, y_in2), up_line[1], bott_line[0]] 
								vehicle_VA = np.array(vehicle_VA)
								cv2.drawContours(direction_img,[vehicle_VA],0,(0,0,255),2)
				
							if len(vehicle_VA) > 0:
								x,y = bott_left
								dist1 = cv2.pointPolygonTest(vehicle_VA.astype(int), (x, y), True)
								if dist1 >= ZONE_DIST_THRESH:
									flag = FLAG_TRUE

								x,y = bott_right
								dist1 = cv2.pointPolygonTest(vehicle_VA.astype(int), (x, y), True)
								if dist1 >= ZONE_DIST_THRESH:
									flag = FLAG_TRUE
								
								x,y = ground_point
								dist1 = cv2.pointPolygonTest(vehicle_VA.astype(int), (x, y), True)
								if dist1 >= ZONE_DIST_THRESH:
									flag = FLAG_TRUE

								# set warning event
								if flag == FLAG_TRUE:
									global_var.global_car_track_dict[id][EVENT_TYPE_IND][0] = 1
									cv2.rectangle(frame_test_vis_table, (int(5*x_off)+3, int(row_cnt*y_off)+3), (int(5*x_off+x_off)-3, int(row_cnt*y_off+y_off)-3), COLOR_PINK, FILL_SHAPE)
								else:
									global_var.global_car_track_dict[id][EVENT_TYPE_IND][0] = 0
									cv2.rectangle(frame_test_vis_table, (int(5*x_off)+3, int(row_cnt*y_off)+3), (int(5*x_off+x_off)-3, int(row_cnt*y_off+y_off)-3), COLOR_GREEN, FILL_SHAPE)

		if global_var.global_car_track_dict[id][EVENT_TYPE_IND][0] == 0 and len(frame_test_car_in_roi_boxes) > 0:
			cv2.rectangle(frame_test_vis_table, (int(5*x_off)+3, int(row_cnt*y_off)+3), (int(5*x_off+x_off)-3, int(row_cnt*y_off+y_off)-3), COLOR_GREEN, FILL_SHAPE)

		row_cnt += 1
	return row_cnt

def update_global_ped_tracking_with_movement(frame_test_vis, frame_test, frame_test_inter, frame_test_vis_table, direction_img, frame_test_car_in_roi_boxes, frame_test_ped_in_roi_boxes, row_cnt):
	# for table visulaization
	x_off = frame_test.shape[0] / 6
	y_off = frame_test.shape[0] / 10

	for id in global_var.global_ped_track_dict:
		track_id_data = global_var.global_ped_track_dict[id]
		ped_moving_p = track_id_data[MOV_POINTS_IND]
		ped_moving_r = track_id_data[MOV_RECT_IND]
		id_counter_ped = track_id_data[OBJ_CNT_IND][0]

		dX_ped = 0
		dY_ped = 0
		direction_ped = ""
		(dirX, dirY) = ("", "")
		exact_angel = None
		disp_x = 0
		disp_y = 0

		for i in np.arange(1, len(ped_moving_p)):
			# if either of the tracked points are None, ignore
			# them
			if ped_moving_p[i - 1] is None or ped_moving_p[i] is None:
				continue
			
			# check object moving or not
			if id_counter_ped >= N_FRAMES_FOR_MOV_OBJ and i == 1 and ped_moving_p[-N_FRAMES_FOR_MOV_OBJ] is not None:
				is_move_x = ped_moving_p[-N_FRAMES_FOR_MOV_OBJ][0] - ped_moving_p[i][0]
				is_move_y = ped_moving_p[-N_FRAMES_FOR_MOV_OBJ][1] - ped_moving_p[i][1]

				if (np.abs(is_move_x) > 0) or (np.abs(is_move_y) > 0):
					global_var.global_ped_track_dict[id][MOV_VOTE_IND].appendleft(1)
				else:
					global_var.global_ped_track_dict[id][MOV_VOTE_IND].appendleft(0)

			if id_counter_ped >= N_FRAMES_FOR_DIR_EST and i == 1 and ped_moving_p[-N_FRAMES_FOR_DIR_EST] is not None:
				move_x = ped_moving_p[-N_FRAMES_FOR_DIR_EST][0] - ped_moving_p[i][0]
				move_y = ped_moving_p[-N_FRAMES_FOR_DIR_EST][1] - ped_moving_p[i][1]
				
				if np.abs(move_x) > 1:
					disp_x = move_x

				if np.abs(move_y) > 1:
					disp_y = move_y
				
			if id_counter_ped >= N_FRAMES_FOR_DIR_EST and i == 1 and ped_moving_p[-N_FRAMES_FOR_DIR_EST] is not None:
				# compute the difference between the x and y
				# coordinates and re-initialize the direction
				# text variables
				dX_ped = ped_moving_p[-N_FRAMES_FOR_DIR_EST][0] - ped_moving_p[i][0]
				dY_ped = ped_moving_p[-N_FRAMES_FOR_DIR_EST][1] - ped_moving_p[i][1]

				(dirX, dirY) = ("", "")

				# ensure there is significant movement in the
				# x-direction
				if np.abs(dX_ped) >= 5:
					dirX = "East" if np.sign(dX_ped) == 1 else "West"
				# ensure there is significant movement in the
				# y-direction
				if np.abs(dY_ped) >= 5:
					dirY = "North" if np.sign(dY_ped) == 1 else "South"
				# handle when both directions are non-empty
				if dirX != "" and dirY != "":
					direction_ped = "{}-{}".format(dirY[:2], dirX)
				# otherwise, only one direction is non-empty
				else:
					direction_ped = dirX if dirX != "" else dirY

			thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
			color = global_var.unique_colors[-global_var.global_ped_track_dict[id][UNI_ID_IND][0]%100]
			cv2.line(frame_test_vis, ped_moving_p[i - 1], ped_moving_p[i], color, thickness)
			cv2.line(frame_test, ped_moving_p[i - 1], ped_moving_p[i], color, thickness)
			cv2.line(frame_test_inter, ped_moving_p[i - 1], ped_moving_p[i], color, thickness)

		color = global_var.unique_colors[-global_var.global_ped_track_dict[id][UNI_ID_IND][0]%100]
		x11, y11, x22, y22 = ped_moving_r[0]
		cv2.rectangle(direction_img, (int(x11), int(y11)), (int(x22), int(y22)), color, THICKNESS_2)

		if np.abs(disp_x) > 1 or np.abs(disp_y) > 1:
			los, sight1_line, sight2_line = find_angel_bet_2_lines(direction_img, x11, y11, x22, y22, disp_x, disp_y)
			global_var.global_ped_track_dict[id][LOS_IND][0] = los
			global_var.global_ped_track_dict[id][LOS_IND][1] = sight1_line
			global_var.global_ped_track_dict[id][LOS_IND][2] = sight2_line
		else:
			global_var.global_ped_track_dict[id][LOS_IND][0] = None
			global_var.global_ped_track_dict[id][LOS_IND][1] = None
			global_var.global_ped_track_dict[id][LOS_IND][2] = None
		
		if id_counter_ped < N_FRAMES_FOR_MOV_OBJ:

			cv2.putText(frame_test_vis_table, "Ped" , (int(0*x_off)+5,int (row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

			cv2.putText(frame_test_vis_table, direction_ped , (int(2*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

		cv2.putText(frame_test_vis_table, "{},{}".format(dX_ped, dY_ped) , (int(3*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

		cv2.circle(frame_test_vis_table, (int(1*x_off + (x_off/2))-5, int(row_cnt*y_off+ (y_off/2)-5)), RADIOUS_5, color, FILL_SHAPE)

		if id_counter_ped >= N_FRAMES_FOR_MOV_OBJ:
			move_vote = 0
			for ped_i in global_var.global_ped_track_dict[id][3]:
				if ped_i == 1:
					move_vote += 1

			if move_vote > MOVING_VOTE_THRESH:
				x11, y11, x22, y22 = ped_moving_r[0]
				mask1 = cv2.rectangle((frame_test_inter[:,:,1]).astype("uint8"), (x11, y11), (x22, y22), (75), FILL_SHAPE)
				frame_test_inter[:,:,1] = (frame_test_inter[:,:,1]).astype("uint8") | mask1

				cv2.putText(frame_test_vis_table, "Ped" , (int(0*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, direction_ped , (int(2*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, "moving" , (int(4*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)
			
			else:
				cv2.putText(frame_test_vis_table, "Ped" , (int(0*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, direction_ped , (int(2*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, "not moving" , (int(4*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_RED, THICKNESS_2, cv2.LINE_AA)

		# if len(frame_test_car_in_roi_boxes) > 0 and len(frame_test_ped_in_roi_boxes) > 0:
		# 	for box in frame_test_ped_in_roi_boxes:
		# 		x11, y11, x22, y22, box_id = box
		# 	if box_id == id:
		# 		# setting waning risk event for car
		# 		# global_var.global_ped_track_dict[id][6][0] = 1
		# 		# line of sight update
		# 		global_var.global_ped_track_dict[id][7][0] = los
		# 		global_var.global_ped_track_dict[id][7].append(sight1_line)
		# 		global_var.global_ped_track_dict[id][7].append(sight2_line)
		row_cnt += 1

	return row_cnt
