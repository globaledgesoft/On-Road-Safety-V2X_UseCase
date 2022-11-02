
from collections import deque
from random import randint

# global parameters
list_points = list()
unique_colors = []
for i in range(100):
    b,g,r = (randint(0, 255), randint(0, 255), randint(0, 255))
    unique_colors.append((b,g,r))

# prepare multi object tracker
model_spec = {'order_pos': 1, 'dim_pos': 2, \
            'order_size': 0, 'dim_size': 2, \
            'q_var_pos': 5000., 'r_var_pos': 0.1}
dt = 1 / 15.0  #fps
# for vehicle
global_car_track_dict = dict()
unique_id_car = 0
# for pedstrian
global_ped_track_dict = dict()
unique_id_ped = 0
# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
car_moving_points = deque(maxlen=32)
ped_moving_points = deque(maxlen=32)
car_moving_rect = deque(maxlen=32)
ped_moving_rect = deque(maxlen=32)

strides = "32,16,8"
anchors = "12,16 19,36 40,28 36,75 76,55 72,146 142,110 192,243 459,401"
mask = "6,7,8 3,4,5 0,1,2"
