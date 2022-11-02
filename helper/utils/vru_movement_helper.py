import global_var 
import cv2
import numpy as np


def draw_directions(direction_img, x11, y11, x22, y22, angle, color):
		length = x22-x11
		x1, y1 = (x11 + ((x22-x11)/2), y11 + ((y22-y11)/2))
		x1, y1 = int(x1), int(y1)
		x2 =  int(round(x1 + length * np.cos(angle * 3.14 / 180.0)))
		y2 =  int(round(y1 + length * np.sin(angle * 3.14 / 180.0)))
		cv2.line(direction_img, (x1,y1), (x2,y2), color, 2)
		line = [(x1,y1), (x2,y2)]
		return line

def find_angel_bet_2_lines(direction_img, x11, y11, x22, y22, disp_x, disp_y):

	x1,y1 = x22, y11 + ((y22-y11)/2) #p0
	x1,y1 = int(x1), int(y1)
	p0 = [x1,y1]

	x2,y2 = (x11 + (x22-x11)/2), y11 + ((y22-y11)/2) #p1
	x2,y2 = int(x2), int(y2)
	p1 = [x2,y2]

	x3, y3 = x11 + ((x22-x11)/2) - disp_x, (y11 + ((y22-y11)/2)) - disp_y #p2
	x3, y3 = int(x3), int(y3)
	p2 = [x3, y3]

	v0 = np.array(p0) - np.array(p1)
	v1 = np.array(p2) - np.array(p1)

	theta = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
	angel = np.degrees(theta)
	draw_directions(direction_img, x11, y11, x22, y22, angel, (0,255,0))
	side_angel1 = angel - 45
	if side_angel1 <= -180:
		diff = side_angel1 - (-180)
		side_angel1 = 180 + diff

	side_angel2 = angel + 45
	if side_angel2 > 180:
		diff = side_angel2 - 180
		side_angel2 = -180 + diff

	draw_directions(direction_img, x11, y11, x22, y22, side_angel1, (255,0,0))
	draw_directions(direction_img, x11, y11, x22, y22, side_angel2, (255,0,0))

	los = (x2, y2), (x3, y3)

	return los

def update_global_car_tracking_with_movement(frame_test_vis, frame_test, frame_test_inter, frame_test_vis_table, direction_img, frame_test_car_in_roi_boxes, frame_test_ped_in_roi_boxes, row_cnt):

	x_off = frame_test.shape[0] / 6
	y_off = frame_test.shape[0] / 10

	for id in global_var.global_car_track_dict:
		track_id_data = global_var.global_car_track_dict[id]
		
		car_moving_p = track_id_data[0]
		car_moving_r = track_id_data[1]
		id_counter_car = track_id_data[2][0]

		dX_car = 0
		dY_car = 0
		direction_car = ""
		(dirX, dirY) = ("", "")

		for i in np.arange(1, len(car_moving_p)):
			# if either of the tracked points are None, ignore
			# them
			if car_moving_p[i - 1] is None or car_moving_p[i] is None:
				continue
			# check to see if enough points have been accumulated in
			# the buffer

			# check object moving or not
			if id_counter_car >= 5 and i == 1 and car_moving_p[-5] is not None:
				is_move_x = car_moving_p[-5][0] - car_moving_p[i][0]
				is_move_y = car_moving_p[-5][1] - car_moving_p[i][1]


				if (np.abs(is_move_x) > 0) or (np.abs(is_move_y) > 0):
					global_var.global_car_track_dict[id][3].appendleft(1)
				else:
					global_var.global_car_track_dict[id][3].appendleft(0)
			
			if id_counter_car >= 10 and i == 1 and car_moving_p[-10] is not None:
				# compute the difference between the x and y
				# coordinates and re-initialize the direction
				# text variables
				dX_car = car_moving_p[-10][0] - car_moving_p[i][0]
				dY_car = car_moving_p[-10][1] - car_moving_p[i][1]

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
			color = global_var.unique_colors[global_var.global_car_track_dict[id][4][0]%100]
			cv2.line(frame_test_vis, car_moving_p[i - 1], car_moving_p[i], color, thickness)
			cv2.line(frame_test, car_moving_p[i - 1], car_moving_p[i], color, thickness)
			cv2.line(frame_test_inter, car_moving_p[i - 1], car_moving_p[i], color, thickness)

		color = global_var.unique_colors[global_var.global_car_track_dict[id][4][0]%100]
		if id_counter_car < 5:

			cv2.putText(frame_test_vis_table, "Car" , (int(0*x_off)+5,int (row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

			cv2.putText(frame_test_vis_table, direction_car , (int(2*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)


		cv2.putText(frame_test_vis_table, "{},{}".format(dX_car, dY_car) , (int(3*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

		cv2.circle(frame_test_vis_table, (int(1*x_off + (x_off/2))-5, int(row_cnt*y_off+ (y_off/2)-5)), 5, color, -1)

		x11, y11, x22, y22 = car_moving_r[0]
		cv2.rectangle(direction_img, (int(x11), int(y11)), (int(x22), int(y22)), color, 2)
		
		if direction_car == "West":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))
			
			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))


		if direction_car == "East":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
		if direction_car == "South":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))


		if direction_car == "North":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))
			

		if direction_car == "No-West":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))
			
		if direction_car == "No-East":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
		if direction_car == "So-West":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

	
		if direction_car == "So-East":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

		if id_counter_car >= 5:
			move_vote = 0
			for car_i in global_var.global_car_track_dict[id][3]:
				if car_i == 1:
					move_vote += 1

			print(id[:5], " => car: ", global_var.global_car_track_dict[id][3], " vot: ", move_vote)

			if move_vote > 3:
				x11, y11, x22, y22 = car_moving_r[0]
				mask1 = cv2.rectangle((frame_test_inter[:,:,1]).astype("uint8"), (x11, y11), (x22, y22), (75), -1)
				frame_test_inter[:,:,1] = (frame_test_inter[:,:,1]).astype("uint8") | mask1

				print(id[:5], " => car is moving: ", global_var.global_car_track_dict[id][3], " vot: ", move_vote, " => ", direction_car)

				cv2.putText(frame_test_vis_table, "Car" , (int(0*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, direction_car , (int(2*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, "moving" , (int(4*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
			else:
				print(id[:5], " => car not moving: ", global_var.global_car_track_dict[id][3], " vot: ", move_vote, " => ", direction_car)

				cv2.putText(frame_test_vis_table, "Car" , (int(0*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, direction_car , (int(2*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, "not moving" , (int(4*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2, cv2.LINE_AA)

		if len(frame_test_car_in_roi_boxes) > 0 and len(frame_test_ped_in_roi_boxes) > 0:
			for box in frame_test_car_in_roi_boxes:
				x11, y11, x22, y22, box_id = box
				# cv2.rectangle(frame_test_inter, (x11, y11), (x22, y22), (255,255,0), 2)
			if box_id == id:
				cv2.rectangle(frame_test_vis_table, (int(5*x_off)+3, int(row_cnt*y_off)+3), (int(5*x_off+x_off)-3, int(row_cnt*y_off+y_off)-3), (0,0,255), -1)
				# setting waning risk event for car
				global_var.global_car_track_dict[id][6][0] = 1

		row_cnt += 1
	return row_cnt

def update_global_ped_tracking_with_movement(frame_test_vis, frame_test, frame_test_inter, frame_test_vis_table, direction_img, frame_test_car_in_roi_boxes, frame_test_ped_in_roi_boxes, row_cnt):

	x_off = frame_test.shape[0] / 6
	y_off = frame_test.shape[0] / 10

	for id in global_var.global_ped_track_dict:
		track_id_data = global_var.global_ped_track_dict[id]
		
		ped_moving_p = track_id_data[0]
		ped_moving_r = track_id_data[1]
		id_counter_ped = track_id_data[2][0]

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
			if id_counter_ped >= 5 and i == 1 and ped_moving_p[-5] is not None:
				is_move_x = ped_moving_p[-5][0] - ped_moving_p[i][0]
				is_move_y = ped_moving_p[-5][1] - ped_moving_p[i][1]

				if (np.abs(is_move_x) > 0) or (np.abs(is_move_y) > 0):
					global_var.global_ped_track_dict[id][3].appendleft(1)
				else:
					global_var.global_ped_track_dict[id][3].appendleft(0)

			if id_counter_ped >= 10 and i == 1 and ped_moving_p[-10] is not None:
				move_x = ped_moving_p[-10][0] - ped_moving_p[i][0]
				move_y = ped_moving_p[-10][1] - ped_moving_p[i][1]
				
				if np.abs(move_x) > 1:
					disp_x = move_x

				if np.abs(move_y) > 1:
					disp_y = move_y
				
					
			if id_counter_ped >= 10 and i == 1 and ped_moving_p[-10] is not None:
				# compute the difference between the x and y
				# coordinates and re-initialize the direction
				# text variables
				dX_ped = ped_moving_p[-10][0] - ped_moving_p[i][0]
				dY_ped = ped_moving_p[-10][1] - ped_moving_p[i][1]

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
			color = global_var.unique_colors[-global_var.global_ped_track_dict[id][4][0]%100]
			cv2.line(frame_test_vis, ped_moving_p[i - 1], ped_moving_p[i], color, thickness)
			cv2.line(frame_test, ped_moving_p[i - 1], ped_moving_p[i], color, thickness)
			cv2.line(frame_test_inter, ped_moving_p[i - 1], ped_moving_p[i], color, thickness)

		color = global_var.unique_colors[-global_var.global_ped_track_dict[id][4][0]%100]
		x11, y11, x22, y22 = ped_moving_r[0]
		cv2.rectangle(direction_img, (int(x11), int(y11)), (int(x22), int(y22)), color, 2)

		# if disp_x > 1 or disp_y > 1:
		# 	los = find_angel_bet_2_lines(direction_img, x11, y11, x22, y22, disp_x, disp_y)
		if direction_ped == "West":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))
			
			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))


		if direction_ped == "East":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
		if direction_ped == "South":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))


		if direction_ped == "North":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))
			

		if direction_ped == "No-West":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))
			
		if direction_ped == "No-East":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
		if direction_ped == "So-West":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

	
		if direction_ped == "So-East":
			angle = 0
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = 90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = 135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,255,0))

			angle = 180
			draw_directions(direction_img, x11, y11, x22, y22, angle, (255,0,0))

			angle = -135
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

			angle = -90
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))
			
			angle = -45
			draw_directions(direction_img, x11, y11, x22, y22, angle, (0,0,255))

		if id_counter_ped < 5:

			cv2.putText(frame_test_vis_table, "Ped" , (int(0*x_off)+5,int (row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

			cv2.putText(frame_test_vis_table, direction_ped , (int(2*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

		print(id[:5], "=> ", "dx: ", dX_ped, " dy: ", dY_ped)

		cv2.putText(frame_test_vis_table, "{},{}".format(dX_ped, dY_ped) , (int(3*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

		cv2.circle(frame_test_vis_table, (int(1*x_off + (x_off/2))-5, int(row_cnt*y_off+ (y_off/2)-5)), 5, color, -1)


		if id_counter_ped >= 5:
			move_vote = 0
			for ped_i in global_var.global_ped_track_dict[id][3]:
				# print(car_i)
				if ped_i == 1:
					move_vote += 1

			print(id[:5], " => ped: ", global_var.global_ped_track_dict[id][3], " vot: ", move_vote)

			if move_vote > 3:
				x11, y11, x22, y22 = ped_moving_r[0]
				mask1 = cv2.rectangle((frame_test_inter[:,:,1]).astype("uint8"), (x11, y11), (x22, y22), (75), -1)
				frame_test_inter[:,:,1] = (frame_test_inter[:,:,1]).astype("uint8") | mask1

				print(id[:5], " => ped is moving: ", global_var.global_ped_track_dict[id][3], " vot: ", move_vote, " => ", direction_ped)

				cv2.putText(frame_test_vis_table, "Ped" , (int(0*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, direction_ped , (int(2*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, "moving" , (int(4*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
			
			else:
				cv2.putText(frame_test_vis_table, "Ped" , (int(0*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, direction_ped , (int(2*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

				cv2.putText(frame_test_vis_table, "not moving" , (int(4*x_off)+5, int(row_cnt*y_off)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2, cv2.LINE_AA)
				
				print(id[:5], " => ped not moving: ", global_var.global_ped_track_dict[id][3], " vot: ", move_vote, " => ", direction_ped)

		print("-----------------")
		# if len(frame_test_car_in_roi_boxes) > 0 and len(frame_test_ped_in_roi_boxes) > 0:
		# 	for box in frame_test_ped_in_roi_boxes:
		# 		x11, y11, x22, y22, box_id = box
		# 		# cv2.rectangle(frame_test_inter, (x11, y11), (x22, y22), (255,255,0), 2)
		# 	if box_id == id:
		# 		# cv2.rectangle(frame_test_vis_table, (int(5*x_off)+3, int(row_cnt*y_off)+3), (int(5*x_off+x_off)-3, int(row_cnt*y_off+y_off)-3), (0,0,255), -1)
		# 		# setting waning risk event for car
		# 		global_var.global_ped_track_dict[id][6][0] = 1
		row_cnt += 1

	return row_cnt
