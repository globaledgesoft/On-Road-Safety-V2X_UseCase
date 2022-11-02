BEV_CORNER_COORDS = 4

IOU_THRESHOLD = 0.4
SCORE_THRESHOLD = 0.3
MAX_OUTPUTS = 100

CAM_FID = 0

CONTOUR_IDX = 0
THICKNESS_2 = 2
THICKNESS_1 = 1
COLOR_BLUE = (255,0,0)
COLOR_GREEN = (0,255,0)
COLOR_RED = (0,0,255)
COLOR_YELLOW = (0,255,255)
COLOR_PINK = (255,0,255)
COLOR_ORANGE = (0,100,255)
TEXT_ORIGIN = (20,20)
RADIOUS_3 = 3
RADIOUS_5 = 5
RADIOUS_9 = 9
FILL_SHAPE = -1

BIG_OBJ_IDX = 0
MED_OBJ_IDX = 1
SMALL_OBJ_IDX = 2

FLAG_TRUE = 1
FLAG_FALSE = 0

#0 car_moving_points
MOV_POINTS_IND = 0
#1 car_moving_rect
MOV_RECT_IND = 1
#2 car_count
OBJ_CNT_IND = 2
#3 move_car_votes
MOV_VOTE_IND = 3
#4 unique_id_car
UNI_ID_IND = 4
#5 in zone flag
IN_ZONE_IND = 5
#6 risk level
EVENT_TYPE_IND = 6
#7 rospy_time
TIME_IND = 7
#8 line of sight
LOS_IND = 8
#9 currunt ground points list
G_POINTS_IND = 9

QUEUE_LEN_32 = 32
PI = 3.14
DEG_180 = 180.0
OBJ_VIEW_ANGLE = 45

N_FRAMES_FOR_MOV_OBJ = 5
N_FRAMES_FOR_DIR_EST = 10
MOVING_VOTE_THRESH = 3
ZONE_DIST_THRESH = -4