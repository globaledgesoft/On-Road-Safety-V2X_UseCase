import numpy as np
# from motpy.motpy.core import Detection
import cv2

def event_table_visulizer(frame, frame_test_vis_table):
	x_off = frame.shape[0] / 6
	y_off = frame.shape[0] / 10

	for r in range(10):
		start_point = (int(r*x_off), 0)
		end_point = (int(r*x_off), frame.shape[0])
		cv2.line(frame_test_vis_table, start_point, end_point, (0,255,0), 1)

		start_point = (0, int(r*y_off))
		end_point = (frame.shape[1], int(r*y_off))
		cv2.line(frame_test_vis_table, start_point, end_point, (0,255,0), 1)

		if r == 0:
			cv2.putText(frame_test_vis_table, "OBJECT" , (int(r*x_off)+5, 0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

		if r == 1:
			cv2.putText(frame_test_vis_table, "ID" , (int(r*x_off+(x_off/2))-5, 0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
			
		
		if r == 2:
			cv2.putText(frame_test_vis_table, "DIRECTION" , (int(r*x_off)+5, 0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2, cv2.LINE_AA)

		if r == 3:
			cv2.putText(frame_test_vis_table, "(Dx, Dy)" , (int(r*x_off)+5, 0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

		if r == 4:
			cv2.putText(frame_test_vis_table, "STATUS" , (int(r*x_off)+5, 0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

		if r == 5:
			cv2.putText(frame_test_vis_table, "EVENT" , (int(r*x_off)+5, 0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

		# return frame_test_vis_table


def get_car_ped_box_detection(boxes,scores,classes):
	array_car_boxes = []
	array_ped_boxes = []

	for i in range(len(classes)):
		if int(classes[i]) == 2 or int(classes[i]) == 5 or int(classes[i]) == 7:
			# array_car_boxes.append((float(boxes[i][0]),float(boxes[i][1]),float(boxes[i][2]),float(boxes[i][3]), scores[i]))
			array_car_boxes.append([float(boxes[i][0]),float(boxes[i][1]),float(boxes[i][2]),float(boxes[i][3])])
			
		elif int(classes[i]) == 0:
			# array_ped_boxes.append((float(boxes[i][0]),float(boxes[i][1]),float(boxes[i][2]),float(boxes[i][3]), scores[i]))
			array_ped_boxes.append([float(boxes[i][0]),float(boxes[i][1]),float(boxes[i][2]),float(boxes[i][3])])
			
	return array_car_boxes, array_ped_boxes

def compute_iou(b_box, b_boxes):
		"""
		b_box: bounding box of shape (4,) 
		b_boxes: ground truth bboxes of shape (n,4)
		"""
		# Compute the co-ordinates of intersected area
		# Upper left corner of intersected area
		x1_I = np.maximum(b_box[0], b_boxes[:, 0])
		y1_I = np.maximum(b_box[1], b_boxes[:, 1])

		# Lower right corner of intersected area
		x2_I = np.minimum(b_box[2], b_boxes[:, 2])
		y2_I = np.minimum(b_box[3], b_boxes[:, 3])

		# Compute the area of intersection
		intersected_area = np.maximum(0, (x2_I - x1_I + 1)) * np.maximum(0, (y2_I - y1_I + 1))

		# Compute b_box area and b_boxes area
		b_box_area = (b_box[2] - b_box[0] + 1) * (b_box[3] - b_box[1] + 1)
		b_boxes_area = (b_boxes[:, 2] - b_boxes[:, 0] + 1) * (b_boxes[:, 3] - b_boxes[:, 1] + 1)

		# Compute IOU
		IOU = intersected_area / (b_box_area + b_boxes_area - intersected_area)

		# return IOU
		return IOU


def get_centroids_and_groundpoints(array_boxes_detected):
	"""
	For every bounding box, compute the centroid and the point located on the bottom center of the box
	@ array_boxes_detected : list containing all our bounding boxes 
	"""
	array_centroids,array_groundpoints = [],[] # Initialize empty centroid and ground point lists 
	for index,box in enumerate(array_boxes_detected):
		# Draw the bounding box 
		# c
		# Get the both important points
		centroid,ground_point = get_points_from_box(box)
		# array_centroids.append(centroid)
		top_left = (box[0], box[1])
		top_right = (box[2], box[1])

		bott_left = (box[0], box[3])
		bott_right = (box[2], box[3])

		array_centroids.append(top_left)
		array_centroids.append(top_right)
		array_centroids.append(bott_left)
		array_centroids.append(bott_right)
		array_centroids.append(centroid)
		array_centroids.append(ground_point)

	return array_centroids,array_centroids


def get_centroids_and_groundpoints_car(box):
	"""
	For every bounding box, compute the centroid and the point located on the bottom center of the box
	@ array_boxes_detected : list containing all our bounding boxes 
	"""
	array_centroids,array_groundpoints = [],[] # Initialize empty centroid and ground point lists 
	# Draw the bounding box 
	# c
	# Get the both important points
	centroid,ground_point = get_points_from_box(box)
	# array_centroids.append(centroid)
	top_left = (box[0], box[1])
	top_right = (box[2], box[1])

	bott_left = (box[0], box[3])
	bott_right = (box[2], box[3])

	array_centroids.append(top_left)
	array_centroids.append(top_right)
	array_centroids.append(bott_left)
	array_centroids.append(bott_right)
	array_centroids.append(centroid)
	array_centroids.append(ground_point)

	return array_centroids,array_centroids


def get_points_from_box(box):
	"""
	Get the center of the bounding and the point "on the ground"
	"""
	center_x = int(((box[0]+box[2])/2))
	center_y = int(((box[1]+box[3])/2))
	# Coordiniate on the point at the bottom center of the box
	center_y_ground = center_y + ((box[3] - box[1])/2)

	return (center_x,center_y),(center_x,int(center_y_ground))

# def process_car_boxes_for_track(bb):
# 	# convert output from OpenCV detector to tracker expected format [xmin, ymin, xmax, ymax]
# 	out_detections = []
# 	for i in range(len(bb)):
# 		xmin, ymin, xmax, ymax, confidence = bb[i]
# 		out_detections.append(Detection(box=[xmin, ymin, xmax, ymax], score=confidence))
# 	return out_detections


def post_process_false_box(boxes):
		bb = boxes.copy()
		cnt = 0
		for i in range(len(boxes)-1):
			ious = compute_iou(boxes[i], np.array(boxes)[i+1:])
			for j in range(len(ious)):
				if ious[j] > 0.8:
					# add score
					x11, y11, x21, y21 = boxes[i]
					x12, y12, x22, y22 = boxes[i+1+j]
					
					bb.remove(bb[i+1+j-cnt])
					cnt += 1
		return bb

# def process_ped_boxes_for_track(bb):
# 	# convert output from OpenCV detector to tracker expected format [xmin, ymin, xmax, ymax]
# 	out_detections = []
# 	for i in range(len(bb)):
# 		xmin, ymin, xmax, ymax, confidence = bb[i]
# 		out_detections.append(Detection(box=[xmin, ymin, xmax, ymax], score=confidence))
		
# 	return out_detections
