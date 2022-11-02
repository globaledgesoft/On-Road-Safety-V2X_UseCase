import numpy as np

# generate different colors for bounding boxes of diffrent classes
COLORS = np.random.uniform(0,255, size=(100, 3))

# ----------------------multitracking setting----------------------
# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

max_age = 4  # no.of consecutive unmatched detection before 
             # a track is deleted

min_hits =3  # no. of consecutive matches needed to establish a track

car_tracker_list =[] # list for trackers
ped_tracker_list =[]
# list for track ID
track_id_list= 0 
debug = False
