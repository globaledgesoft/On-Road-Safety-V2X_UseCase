import cv2
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from linear_assignment import  linear_assignment
import utils
import KalmanFilter as kf_tracker
from settings import *

car_tracker_list =[] # list for trackers
# ped_tracker_list = []
car_track_id_list = 0
ped_track_id_list = 0


def Get_tracker_list():
    return Get_tracker_list

def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    if len(bb_test)==0 or len(bb_gt)==0:
        return None

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    #print(bb_test)
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    # #print(type(yy2))
    # yy1=[[70]]
    # xx2=[[334]]
    # try:
    #     yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    # except:
    #     yy2=np.array([153.,154.,154.,154.,154.])    
    # w = np.maximum(0., xx2 - xx1)
    # if  w[0]==0:
    #     w=[[33.4942]]

    # h = np.maximum(0., yy2 - yy1)
    # print("w,h",w,h)
    # wh = w * h
    # if wh[0][0]==0:
    #     wh=[[2709]]

    #print("yy1_shape",yy1.shape)
    # print("w",w)
    # print("h",h)
    # print("wh",wh)
    # print("w,h",w,h)

    # if yy1.size > 0:
    #     print("yy1",yy1[0][0])
    # else:
    #     print('array has a size of 0')

    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)  

def associate_detections_to_trackers(trackers, detections,iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)
    if iou_matrix is None:
        return np.array([]),np.array([]),np.array([])

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def pipline_car(frame, roi_boxes):
        global car_tracker_list
        global car_track_id_list
        z_box = [x for x in roi_boxes]
        x_box = []

        if len(car_tracker_list) > 0:
            for trk in car_tracker_list:
                x_box.append(trk.box)

        #print("zbox: ", z_box)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(x_box, z_box, iou_threshold = 0.3)  

        # Deal with matched detections     
        if matched.size >0:
            for trk_idx, det_idx in matched:
                if len(z_box) <= det_idx :
                    continue
                z = z_box[det_idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk= car_tracker_list[trk_idx]
                tmp_trk.kalman_filter(z)
                xx = tmp_trk.x_state.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                x_box[trk_idx] = xx
                tmp_trk.box =xx
                tmp_trk.hits += 1
                tmp_trk.no_losses = 0
        
        # Deal with unmatched detections      
        if len(unmatched_dets)>0:
            for idx in unmatched_dets:
                z = z_box[idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk = kf_tracker.Tracker() # Create a new tracker
                x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                tmp_trk.id = car_track_id_list#.popleft() # assign an ID for the tracker
                car_track_id_list+=1
                car_tracker_list.append(tmp_trk)
                x_box.append(xx)
        
        # Deal with unmatched tracks       
        if len(unmatched_trks)>0:
            for trk_idx in unmatched_trks:
                tmp_trk = car_tracker_list[trk_idx]
                tmp_trk.no_losses += 1
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box =xx
                x_box[trk_idx] = xx
                                    
        # Book keeping
        # deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)          
        
        car_tracker_list = [x for x in car_tracker_list if x.no_losses<=max_age]
    
        return car_tracker_list


def pipline_ped(frame, roi_boxes):
        global ped_tracker_list
        global ped_track_id_list
        z_box = [x for x in roi_boxes]
        x_box = []

        if len(ped_tracker_list) > 0:
            for trk in ped_tracker_list:
                x_box.append(trk.box)

        print()
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(x_box, z_box, iou_threshold = 0.3)  

        # Deal with matched detections     
        if matched.size >0:
            for trk_idx, det_idx in matched:
                # print("det_idx",det_idx)
                # print("z_box",z_box)
                # print("len of z box",len(z_box))
                if len(z_box)<=det_idx:
                    continue
                
                # if det_idx==1 or det_idx==2 or det_idx==3:
                #     det_idx=0
                    
                z = z_box[det_idx]
                #print("z",z)

                # if len(z_box[det_idx])==1 :
                #     return None

                z = np.expand_dims(z, axis=0).T
                tmp_trk= ped_tracker_list[trk_idx]
                tmp_trk.kalman_filter(z)
                xx = tmp_trk.x_state.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                x_box[trk_idx] = xx
                tmp_trk.box =xx
                tmp_trk.hits += 1
                tmp_trk.no_losses = 0
        
        # Deal with unmatched detections      
        if len(unmatched_dets)>0:
            for idx in unmatched_dets:
                z = z_box[idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk = kf_tracker.Tracker() # Create a new tracker
                x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                tmp_trk.id = ped_track_id_list#.popleft() # assign an ID for the tracker
                ped_track_id_list+=1
                ped_tracker_list.append(tmp_trk)
                x_box.append(xx)
        
        # Deal with unmatched tracks       
        if len(unmatched_trks)>0:
            for trk_idx in unmatched_trks:
                tmp_trk = ped_tracker_list[trk_idx]
                tmp_trk.no_losses += 1
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box =xx
                x_box[trk_idx] = xx
                                    
        # Book keeping
        # deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)          
        
        ped_tracker_list = [x for x in ped_tracker_list if x.no_losses<=max_age]
    
        return ped_tracker_list
     
