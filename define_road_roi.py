import global_var 
import cv2
import numpy as np

# Define the callback function that we are going to use to get our coordinates
def CallBackFunc(event, x, y, flags, param):
    global_var.list_points
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (", x, ", ",y, ")")
        global_var.list_points.append([x,y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
        global_var.list_points.append([x,y])

def take_view(size_frame, video_name):
    global_var.list_points
    vs = cv2.VideoCapture(video_name)
    # Loop until the end of the video stream
    while True:    
        # Load the frame and test if it has reache the end of the video
        (frame_exists, frame) = vs.read()
        frame = cv2.resize(frame, (size_frame,size_frame))
        cv2.imwrite(video_name[:-3]+"_static_frame_from_video.jpg",frame)
        break
    # Create a black image and a window
    windowName = 'MouseCallback'
    cv2.namedWindow(windowName)
    # Load the image 
    img_path = video_name[:-3]+"_static_frame_from_video.jpg"
    img = cv2.imread(img_path)
    # Get the size of the image for the calibration
    width,height,_ = img.shape
    print(img.shape)
    # Create an empty list of points for the coordinates
    global_var.list_points = list()
    # bind the callback function to window
    cv2.setMouseCallback(windowName, CallBackFunc)
    # Check if the 4 points have been saved
    while (True):
        cv2.imshow(windowName, img)
        if len(global_var.list_points) == BEV_CORNER_COORDS:
            p2 = global_var.list_points[3]
            p1 = global_var.list_points[2]
            p4 = global_var.list_points[0]
            p3 = global_var.list_points[1]
            width_og = width
            height_og = height
            img_path = img_path
            size_frame = size_frame
            corner_points = [p1, p2, p3, p4]
            break
        if cv2.waitKey(20) == 27:
            break
    cv2.destroyAllWindows()
    return corner_points, width_og, height_og, img_path, size_frame

def take_ROI(size_frame, video_name, test_dir, vs):
    global_var.list_points
    # Loop until the end of the video stream
    while True:    
        # Load the frame and test if it has reache the end of the video
        (frame_exists, frame) = vs.read()
        if frame_exists:
            frame = cv2.resize(frame, (size_frame,size_frame))
            cv2.imwrite(video_name[:-3]+"_static_frame_from_video.jpg",frame)
            break
    # Create a black image and a window
    windowName = 'MouseCallback'
    cv2.namedWindow(windowName)
    # Load the image 
    img_path = video_name[:-3]+"_static_frame_from_video.jpg"
    img = cv2.imread(img_path)
    # Get the size of the image for the calibration
    width,height,_ = img.shape
    print(img.shape)
    # Create an empty list of points for the coordinates
    global_var.list_points = list()
    # bind the callback function to window
    cv2.setMouseCallback(windowName, CallBackFunc)
    # Check if the 4 points have been saved
    while (True):
        cv2.imshow(windowName, img)
        k = cv2.waitKey(1)
        if k == ord('c'):
            break
        if k == 27:
            break
    cv2.destroyAllWindows()
    return global_var.list_points
