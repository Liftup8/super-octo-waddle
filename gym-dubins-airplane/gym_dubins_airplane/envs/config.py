<<<<<<< HEAD
import math
import numpy as np


class Config:
    G = 9.8
    EPISODES = 1000
    vel_mps = 20 # velocity of aircrafts
    
    action_time = 0.5
    action_size = 15



    red_health = 0
    blue_health = 0
    # input dim
    window_width = 800  # pixels
    window_height = 800  # pixels
    window_z = 800  # pixels
    diagonal = 800  # this one is used to normalize dist_to_intruder
    tick = 30
    scale = 30
    d_min = 25  # minimum distance between aircrafts for gunfire (dangerous circle)
    d_max = 300  # maximum distance between aircrafts for gunfire (outer circle)

    # distance param
    minimum_separation = 555 / scale
    NMAC_dist = 150 / scale
    horizon_dist = 4000 / scale
    initial_min_dist = 3000 / scale
    goal_radius = 600 / scale

    dist_norm = window_width
    deg_norm = np.pi

    # speed
    min_speed = 50 / scale
    max_speed = 80 / scale
    d_speed = 5 / scale
    speed_sigma = 2 / scale
    position_sigma = 10 / scale

    # maximum training steps
    max_steps = 1000
=======
import math
import numpy as np


class Config:
    G = 9.8
    EPISODES = 1000

    red_health = 0
    blue_health = 0
    # input dim
    window_width = 800  # pixels
    window_height = 800  # pixels
    window_z = 800  # pixels
    diagonal = 800  # this one is used to normalize dist_to_intruder
    tick = 30
    scale = 30
    d_min = 25  # minimum distance between aircrafts for gunfire (dangerous circle)
    d_max = 300  # maximum distance between aircrafts for gunfire (outer circle)

    # distance param
    minimum_separation = 555 / scale
    NMAC_dist = 150 / scale
    horizon_dist = 4000 / scale
    initial_min_dist = 3000 / scale
    goal_radius = 600 / scale

    dist_norm = 800
    deg_norm = np.pi

    # speed
    min_speed = 50 / scale
    max_speed = 80 / scale
    d_speed = 5 / scale
    speed_sigma = 2 / scale
    position_sigma = 10 / scale

    # maximum training steps
    max_steps = 1000

    # reward setting
    position_reward = 10. / 10.
    heading_reward = 10 / 10.

    collision_penalty = -5. / 10
    outside_penalty = -1. / 10

    step_penalty = -0.01 / 10
>>>>>>> 6819adb566d3adb52b4ba0d843df0a5e09f4af63
