import numpy as np

def _nms_boxes(boxes, scores, iou_thresh=0.45):
	x = boxes[:, 0]
	y = boxes[:, 1]
	# w = boxes[:, 2] -  boxes[:, 0]
	# h = boxes[:, 3] -  boxes[:, 1]
	w = boxes[:, 2] 
	h = boxes[:, 3] 

	areas = (w)* (h)
	order = scores.argsort()[::-1]

	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)

		xx1 = np.maximum(x[i], x[order[1:]])
		yy1 = np.maximum(y[i], y[order[1:]])
		xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
		yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

		w1 = np.maximum(0.0, xx2 - xx1 + 1)
		h1 = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w1 * h1

		ovr = inter / (areas[i] + areas[order[1:]] - inter)
		inds = np.where(ovr <= iou_thresh)[0]
		order = order[inds + 1]

	keep = np.array(keep)

	return keep

def sigmoid(x):
	return (1 / (1 + np.exp(-x)))

def post_process_tiny_predictions(inputs, anchors, mask, strides, scale, max_outputs, iou_threshold, score_threshold, ind, num_classes=2):
	
	#dtype = inputs[0].dtype
	logits = inputs[0]
	i = ind
	anchors, stride = anchors[mask[i]], strides[i]
	x_shape = (1, scale, scale, len(anchors), num_classes+5)
	#print (x_shape)
	logits = np.reshape(logits, (1, scale, scale, len(anchors), num_classes + 5))

	box_xy, box_wh, obj, cls = logits[...,:2], logits[...,2:4], logits[...,4], logits[...,5:]
	
	box_xy = sigmoid(box_xy)
	obj = sigmoid(obj)
	#print(obj)
	cls = sigmoid(cls)

	grid_shape = (scale, scale)
	grid_h, grid_w = grid_shape[0], grid_shape[1]
	anchors = np.array(anchors, dtype=np.float32)
	grid = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
	grid = np.expand_dims(np.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

	box_xy = (box_xy + np.array(grid, dtype=np.float32)) * stride
	box_wh = np.exp(box_wh) * np.array(anchors, dtype=np.float32)

	box_x1y1 = box_xy - box_wh / 2.
	box_x2y2 = box_xy + box_wh / 2.
	box = np.concatenate([box_x1y1, box_x2y2], axis=-1)

	all_boxes = np.reshape(box, (x_shape[0], -1, 1, 4))
	objects = np.reshape(obj, (x_shape[0], -1, 1))
	all_classes = np.reshape(cls, (x_shape[0], -1, num_classes))

	all_scores = objects * all_classes
	return all_boxes, all_scores, all_classes

def filter_boxes(all_boxes, all_scores, all_classes, score_thresh=0.2, iou_thresh=0.45):
	
	box_classes = np.argmax(all_scores, axis=-1)
	box_class_scores = np.max(all_scores, axis=-1)
	# print("box_class_scores_shape : ", box_class_scores.shape )
	# print(box_class_scores[0][100:110])
	# print(box_class_scores[0][5000])
	#pos = np.where(box_class_scores >= score_thresh)
	pos = np.where(box_class_scores >= 0.2)
	
	# print(pos)

	fil_boxes = all_boxes[pos]
	fil_boxes = np.reshape(fil_boxes, (fil_boxes.shape[0], fil_boxes.shape[2]))
	fil_classes = box_classes[pos]
	fil_scores = box_class_scores[pos]
	# print("fil_classes : ", fil_classes)

	nboxes, nclasses, nscores = [], [], []
	for c in set(fil_classes):
		inds = np.where(fil_classes == c)
		b = fil_boxes[inds]
		c = fil_classes[inds]
		s = fil_scores[inds]

		keep = _nms_boxes(b, s)
		# print("keep " ,keep)

		nboxes.append(b[keep])
		nclasses.append(c[keep])
		nscores.append(s[keep])
	
	# print("nboxes : ", nboxes)

	if not nclasses and not nscores:
		return None, None, None

	return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


