
import os
import sys
sys.path.append("./lib/")
import lib.qcsnpe as qc
import cv2
import numpy as np
import json
import argparse
import time
from collections import deque
from random import randint
import sys
import aws_config as config 
import json
sys.path.append("/data/ram/libboto3/")
import boto3
from os import path

client= boto3.client('iot-data',aws_access_key_id =config.aws_access_key_id ,aws_secret_access_key =config.aws_secret_access_key , region_name=config.region, endpoint_url=config.endpoint )


from helper.utils.img_helper import _draw_bounding_boxes, preprocess_img, postprocess_img, decode_class_names
from helper.utils.postprocess import filter_boxes, _nms_boxes, sigmoid, post_process_tiny_predictions
from helper.model.yolov4 import my_model as Model
from helper.utils.bird_view_transfo_functions import compute_perspective_transform,compute_point_perspective_transformation
from helper.utils.detection_helper import *
from vru_helper import update_global_car_tracking_list, update_global_ped_tracking_list
from vru_exact_movement_helper import update_global_car_tracking_with_movement, update_global_ped_tracking_with_movement
from helper.utils.constants import *
from define_road_roi import *
import global_var 
import KalmanFilter as kf_tracker
from settings import *
import multitracking
import utils

config_file = open('config.json')
config = json.load(config_file)

CPU = 0
GPU = 1
DSP = 2
class_name_path = "model_data/coco.name" 
test_dir = "video_test/"
cam_flag = 1
video_path = "video_test/p9_ped.ts"
if video_path == None:
    video_path = "./"
    cam_flag = 0
image_size = 416
iou_threshold = 0.5
score_threshold = 0.2
max_outputs = 100
names = decode_class_names(class_name_path)
num_classes = len(names)
strides = global_var.strides
anchors = global_var.anchors
mask = global_var.mask
anchors = np.array(list(map(lambda x: list(map(int, str.split(x, ','))), anchors.split())))
mask = np.array(list(map(lambda x: list(map(int, str.split(x, ','))), mask.split())))
strides = list(map(int, strides.split(',')))
iCounter=0

# to get ROI
def get_vulnerable_region():
    global_var.list_points
    if(path.exists(video_path[:-3]+"_ROI1_vulnerable.json") is False):
        list_points =  take_ROI(image_size, video_path, test_dir, video)
        ROI1_dict = {"list_points":list_points}
        a_file = open(video_path[:-3]+"_ROI1_vulnerable.json", "w")
        json.dump(ROI1_dict, a_file)
        a_file.close()
        print(video_path[:-3]+"_ROI1_vulnerable.json")
    else:
        a_file = open(video_path[:-3]+"_ROI1_vulnerable.json", "r")
        ROI1_dict = json.load(a_file)
        list_points = ROI1_dict["list_points"]
        a_file.close()
   
    r_points_static = []
    for p in list_points:
        x,y = p
        r_points_static.append([x,y])

    r_points_static = np.array(r_points_static)
    return r_points_static

# for postprocessing 
def postprocess(frame,boxes,right_classes,right_scores,frame_cnt,out):
    r_points_static = get_vulnerable_region()
    
    frame = cv2.resize(frame, (image_size,image_size))
    # for visualization
    frame_test = frame.copy().astype("uint8")
    frame_track = frame.copy().astype("uint8")
    frame_test_vis = np.zeros((frame.shape[0], frame.shape[1], 3)).astype("uint8")
    frame_test_vis_table = np.zeros((frame.shape[0], frame.shape[1], 3)).astype("uint8")
    frame_test_inter = frame.copy()
    frame_test_all_result = np.zeros((frame_test.shape[0],frame_test.shape[1]*2,3), np.uint8)
    direction_img = np.zeros((frame.shape[0], frame.shape[1], 3)).astype("uint8")
    event_table_visulizer(frame, frame_test_vis_table)

    cv2.drawContours(frame_test,[r_points_static],CONTOUR_IDX,COLOR_BLUE,THICKNESS_2)
    cv2.drawContours(frame_test_vis,[r_points_static],CONTOUR_IDX,COLOR_BLUE,THICKNESS_2)
    cv2.drawContours(frame_test_inter,[r_points_static],CONTOUR_IDX,COLOR_BLUE,THICKNESS_2)
    cv2.drawContours(direction_img,[r_points_static],CONTOUR_IDX,COLOR_YELLOW,THICKNESS_1)
            
    img, boxes = postprocess_img(frame, frame.shape[1::-1], boxes)

    if boxes is not None:
        
        img = _draw_bounding_boxes(img, boxes, right_scores, right_classes, names)
        # get car and ped boxes
        array_car_boxes, array_ped_boxes = get_car_ped_box_detection(boxes, right_scores, right_classes)
        # for car tracking .........
        array_car_boxes = post_process_false_box(array_car_boxes)
        car_boxes_track = array_car_boxes # kalman tracking
        # print(car_boxes_track)
        car_tracker_list = multitracking.pipline_car(frame_track, car_boxes_track) # kalman tracking
        # The list of tracks to be annotated  
        good_car_tracker_list =[]
        for trk in car_tracker_list:
            if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
                good_car_tracker_list.append(trk)
                x_cv2 = trk.box
                frame_track= utils.draw_box_label(frame_track, x_cv2, trk.id) # Draw the bounding boxes on the  images
                                                # draw line here

        # discard vanished car from record
        for id in global_var.global_car_track_dict.copy():
            rem_id = FLAG_TRUE
            for track in good_car_tracker_list:
                if id == track.id:
                    rem_id = FLAG_FALSE
                    break
            if rem_id == FLAG_TRUE:
                del global_var.global_car_track_dict[id]


        # for ped tracking .........
        ped_boxes_track = array_ped_boxes
        ped_tracker_list = multitracking.pipline_ped(frame_track, ped_boxes_track) # kalman tracking
        # The list of tracks to be annotated  
        good_ped_tracker_list =[]
        for trk in ped_tracker_list:
            if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
                good_ped_tracker_list.append(trk)
                x_cv2 = trk.box
                frame_track= utils.draw_box_label(frame_track, x_cv2, trk.id) # Draw the bounding boxes on the  images
                                                # draw line here

        # discard vanished ped from record
        for id in global_var.global_ped_track_dict.copy():
            rem_id = FLAG_TRUE
            for track in good_ped_tracker_list:
                if id == track.id:
                    rem_id = FLAG_FALSE
                    break
            if rem_id == FLAG_TRUE:
                del global_var.global_ped_track_dict[id]
        
        # car dict update with track id
        frame_test_car_in_roi_boxes =  update_global_car_tracking_list(good_car_tracker_list, frame_cnt, frame_test, frame_test_vis, frame_test_inter, direction_img, r_points_static)
        # ped dict update with track id
        frame_test_ped_in_roi_boxes =  update_global_ped_tracking_list(good_ped_tracker_list, frame_cnt, frame_test, frame_test_vis, frame_test_inter, direction_img, r_points_static)
        row_cnt = 1
        # movement tracking of ped
        row_cnt = update_global_ped_tracking_with_movement(frame_test_vis, frame_test, frame_test_inter, frame_test_vis_table, direction_img, frame_test_car_in_roi_boxes, frame_test_ped_in_roi_boxes, row_cnt)
        # movement tracking of car
        row_cnt = update_global_car_tracking_with_movement(frame_test_vis, frame_test, frame_test_inter, frame_test_vis_table, direction_img, frame_test_car_in_roi_boxes, frame_test_ped_in_roi_boxes, row_cnt)
        
        if len(frame_test_car_in_roi_boxes) > 0 and len(frame_test_ped_in_roi_boxes) > 0:
            for box in frame_test_car_in_roi_boxes:
                x11, y11, x22, y22, _ = box
                cv2.rectangle(frame_test_inter, (x11, y11), (x22, y22), (0,0,255), 2)

            for box in frame_test_ped_in_roi_boxes:
                x11, y11, x22, y22, _ = box
                cv2.rectangle(frame_test_inter, (x11, y11), (x22, y22), (0,0,255), 2)
        i = 0
        for id in global_var.global_ped_track_dict:
            track_id_data = global_var.global_ped_track_dict[id]
            x1, y1, x2, y2 = track_id_data[1][0]
            #print("pedestrian detected", [x1, y1, x2, y2])
            # printing msg on aws
            message_json = {"pedestrian detected":[x1, y1, x2, y2]}
            message = json.dumps(message_json)
            response = client.publish(topic = "qcs610/msg",qos = 1,payload = message)
            if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                print("Info send to AWS")
            
        i = 0
        detection_str = ""
        for id in global_var.global_car_track_dict:
            track_id_data = global_var.global_car_track_dict[id]
            x1, y1, x2, y2 = track_id_data[1][0]
            detection_str += '{"box":' + str([x1, y1, x2, y2])
            detection_str += ',"class":' + "car"
            detection_str += ' ,"isObjectZone":' + str(track_id_data[5][0])
            detection_str += ' ,"EventType":' + str(track_id_data[6][0])
            detection_str += ' ,"track_id":' + str(id)
            detection_str += ' ,"frame_count":' + str(frame_cnt)
            detection_str += ' ,"timestamp": %s}' % "ros_time"
            i += 1
            print("event type: ", str(track_id_data[6][0]))
                   
        
    f_height, f_width, _ = frame_test.shape
    cv2.putText(frame_test_all_result, str(frame_cnt), TEXT_ORIGIN, cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_YELLOW, 2, cv2.LINE_AA)
    frame_test_all_result[0:f_height, 0:f_width, :] =  frame_test_all_result[0:f_height, 0:f_width, :] | frame_track
    frame_test_all_result[0:f_height, f_width:f_width*2, :] = frame_test_all_result[0:f_height, f_width:f_width*2, :] | frame_test_vis_table
    

    frame_test_all_result = cv2.resize(frame_test_all_result, (1280,720))
   #cv2.imshow("frame_test_all_result",frame_test_all_result)
    out.write(frame_test_all_result)

# main function            
def callback():

    cap = cv2.VideoCapture(config['camera_pipeline'], cv2.CAP_GSTREAMER)
    out_layers =np.array(["conv2d_9/Conv2D","conv2d_12/Conv2D"])
    model_path = config['model_path']
    model = qc.qcsnpe(model_path,out_layers, 1)
    #model = qc.qcsnpe("yolo_v3.dlc", out_layers, CPU)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path[:-3]+"_output.avi", fourcc, 9.0,(1280,720))
    frame_cnt=0
    while(video.isOpened()):
        frame_cnt+=1
        ret, frame = video.read()
        if type(frame)==type(None):
            exit()
        orig_img = frame.copy()
        orig_img = cv2.resize(frame,(416,416))
        image=cv2.resize(frame,(image_size,image_size))
                     
        imgs = image.copy()
        output=model.predict(imgs)
        out0 = output["conv2d_9/BiasAdd:0"]
        out1 = output["conv2d_12/BiasAdd:0"]
        
        out0=np.reshape(out0,(1,13,13,3,85))
        out1=np.reshape(out1,(1,26,26,3,85))
        
     
        big_all_boxes, big_all_scores, big_all_classes = post_process_tiny_predictions([out0], anchors, mask, strides, max_outputs, iou_threshold, score_threshold, BIG_OBJ_IDX, num_classes)
        mid_all_boxes, mid_all_scores, mid_all_classes = post_process_tiny_predictions([out1], anchors, mask, strides, max_outputs, iou_threshold, score_threshold, MED_OBJ_IDX, num_classes)
        
        bboxes = np.concatenate([big_all_boxes, mid_all_boxes], axis=1)
        scores = np.concatenate([big_all_scores, mid_all_scores], axis=1)
        classes = np.concatenate([big_all_classes, mid_all_classes], axis=1)

        boxes, classes, scores = filter_boxes(bboxes, scores, classes)
        
        if type(boxes)==type(None):
            continue        
        boxes_len = boxes.shape[0]
        for i in range(boxes_len):
            x1, y1, x2, y2 = (boxes[i]/416.0)
            x1 = int(x1*orig_img.shape[1])
            x2 = int(x2*orig_img.shape[1])
            y1 = int(y1*orig_img.shape[0])
            y2 = int(y2*orig_img.shape[0])

            orig_img = cv2.rectangle(orig_img, (x1,y1), (x2, y2), (255,255,0), 3)
            postprocess(orig_img, boxes, classes, scores,frame_cnt,out)
            
    out.release()
video=cv2.VideoCapture("video_test/p9_ped.ts")


if __name__ == "__main__" :
    callback()
 
