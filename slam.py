# Pratik Chaudhari (pratikac@seas.upenn.edu)
# Minku Kim (minkukim@seas.upenn.edu)

import os, sys, pickle, math
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from load_data import load_kitti_lidar_data, load_kitti_poses, load_kitti_calib
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    def __init__(s, resolution=0.5):
        s.resolution = resolution
        s.xmin, s.xmax = -700, 700
        s.zmin, s.zmax = -500, 900
        # s.xmin, s.xmax = -400, 1100
        # s.zmin, s.zmax = -300, 1200

        s.szx = int(np.ceil((s.xmax - s.xmin) / s.resolution + 1))
        s.szz = int(np.ceil((s.zmax - s.zmin) / s.resolution + 1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szz), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds,
        # and similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh / (1 - s.occupied_prob_thresh))

    def grid_cell_from_xz(s, x, z):
        """
        x and z can be 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/z go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        ##### TODO: XXXXXXXXXX
        x = np.clip(x, s.xmin, s.xmax)
        z = np.clip(z, s.zmin, s.zmax)
        x_idx = np.floor((x - s.xmin) / s.resolution).astype(int)
        z_idx = np.floor((z - s.zmin) / s.resolution).astype(int)
        return np.array([x_idx, z_idx])

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.5, Q=1e-3*np.eye(3), resampling_threshold=0.3):
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

        # dynamics noise for the state (x, z, yaw)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar_dir = src_dir + f'odometry/{s.idx}/velodyne/'
        s.poses = load_kitti_poses(src_dir + f'poses/{s.idx}.txt')
        s.lidar_files = sorted(os.listdir(src_dir + f'odometry/{s.idx}/velodyne/'))
        s.calib = load_kitti_calib(src_dir + f'calib/{s.idx}/calib.txt')

    def init_particles(s, n=100, p=None, w=None):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3, s.n))
        s.w = deepcopy(w) if w is not None else np.ones(n) / n

    @staticmethod
    def stratified_resampling(p, w):
        """
        Resampling step of the particle filter.
        """
        ##### TODO: XXXXXXXXXXX
        r = np.random.uniform(0, 1 / len(w))
        c = w[0]
        i = 0
        new_particles = np.zeros_like(p)
        new_weights = deepcopy(w)
        for m in range(len(w)):
            U = r + m / len(w)
            while U > c:
                i += 1
                c += w[i]
            new_particles[:, m] = p[:, i]
            new_weights[m] = w[i]
        return new_particles, new_weights

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def lidar2world(s, p, points):
        """
        Transforms LiDAR points to world coordinates.

        The particle state p is now interpreted as [x, z, theta], where:
        - p[0]: x translation
        - p[1]: z translation
        - p[2]: rotation in the x-z plane

        The input 'points' is an (N, 3) array of LiDAR points in xyz.
        """
        #### TODO: XXXXXXXXX
        # 1. Convert LiDAR points to homogeneous coordinates
        lidar_hom_coords = make_homogeneous_coords_3d(points.T)

        # 2. Transform Velodyne Frame -> Camera Frame
        Tr = s.calib
        lidar_in_camera_frame = Tr @ lidar_hom_coords

        # 3. from camera frame to world frame.
        R_camera_to_world = np.array([[cos(p[2]), 0, sin(p[2])],
                                        [0, 1, 0],
                                        [-sin(p[2]), 0, cos(p[2])]])
        translation = np.array([p[0], 0, p[1]])
        world_coords = R_camera_to_world @ lidar_in_camera_frame + translation.reshape(-1, 1)
        return world_coords[:3, :].T

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d
        function to get the difference of the two poses and we will simply
        set this to be the control.
        Extracts control in the state space [x, z, rotation] from consecutive poses.
        [x, z, theta]
        theta is the rotation around the Y-axis
              | cos  0  -sin |
        R_y = |  0   1    0  |
              |+sin  0   cos |
        R31 = +sin
        R11 =  cos
        yaw = atan2(R_31, R_11)
        """
        if t == 0:
            return np.zeros(3)

        #### TODO: XXXXXXXXXXX
        T1, T2 = s.poses[t-1], s.poses[t]
        p1 = np.array([T1[0, 3], T1[2, 3], np.arctan2(T1[0, 2], T1[0, 0])])
        p2 = np.array([T2[0, 3], T2[2, 3], np.arctan2(T2[0, 2], T2[0, 0])])
        control = smart_minus_2d(p2, p1)
        return control


    def dynamics_step(s, t):
        """
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter
        """
        #### TODO: XXXXXXXXXXX
        control = s.get_control(t)
        noise_vec = np.random.multivariate_normal(np.zeros(3), s.Q, s.n).T
        for i in range(s.n):
            s.p[:, i] = smart_plus_2d(s.p[:, i], control)
            s.p[:, i] = smart_plus_2d(s.p[:, i], noise_vec[:, i])
        return s.p


    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        log_w = np.log(w)
        updated_weights = log_w + obs_logp
        updated_weights -= slam_t.log_sum_exp(updated_weights)
        return np.exp(updated_weights)
    
    def bresenham2D(self, x1, y1, x2, y2):
        """
        Bresenham's line algorithm.
        Traverse, the grid along the line from (x0, y0) to (x1, y1).
        """
        dx = x2 - x1
        dy = y2 - y1

        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)

        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True

        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1

        # Calculate error
        error = dx // 2
        ystep = 1 if y1 < y2 else -1

        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx

        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()

        return points
    
    def find_free_cells_from_obstacles(s, particle_pose, lidar_in_world_coords):
        """
        This function finds the free cells in the map given the obstacles in the world coordinates.
        The obstacles are the LiDAR points in the world coordinates.
        Free cells are the cells from the particle towards the LiDAR points.
        """
        # Get the grid coordinates of the particle
        grid_coords = s.map.grid_cell_from_xz(particle_pose[0], particle_pose[1])
        free_cells = []
        set_lidar_grid_coords = set()
        for i in range(len(lidar_in_world_coords)):
            lidar_point = lidar_in_world_coords[i]
            # Get the grid coordinates of the LiDAR point
            lidar_grid_coords = s.map.grid_cell_from_xz(lidar_point[0], lidar_point[2])
            if tuple(lidar_grid_coords) in set_lidar_grid_coords:
                continue
            set_lidar_grid_coords.add(tuple(lidar_grid_coords))
            # Get the cells between the particle and the LiDAR point
            cells = s.bresenham2D(grid_coords[0], grid_coords[1], lidar_grid_coords[0], lidar_grid_coords[1])
            free_cells.extend(cells)

        free_cells = np.array(free_cells)
        free_cells = np.unique(free_cells, return_index=False, axis=0)
        return free_cells


    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data
        you can also store a thresholded version of the map here for plotting later
        """

        #### TODO: XXXXXXXXXXX
        # Update particle weights by obtaining log probabilities of observations


        # Get LiDAR points
        lidar_file = s.lidar_files[t]
        lidar_points = load_kitti_lidar_data(s.lidar_dir + lidar_file)
        # remove reflectivity values
        lidar_points = lidar_points[:, :3]

        if t == 0:
            world_coords = s.lidar2world(s.p[:, 0], lidar_points)
            grid_coords = s.map.grid_cell_from_xz(world_coords[:, 0], world_coords[:, 2])
            s.map.cells[grid_coords[0], grid_coords[1]] = 1
            return

        observation_logp = np.zeros(s.n)
        for i in range(s.n):
            world_coords = s.lidar2world(s.p[:, i], lidar_points)
            grid_coords = s.map.grid_cell_from_xz(world_coords[:, 0], world_coords[:, 2])
            # Get log-odds of occupied cells
            observation_logp[i] = np.sum(s.map.cells[grid_coords[0], grid_coords[1]])

        # Update weights
        s.w = s.update_weights(s.w, observation_logp)

        # Update map using the maximum weight particle
        max_weight_idx = np.argmax(s.w)
        chosen_particle = s.p[:, max_weight_idx]
        lidar_in_world_coords = s.lidar2world(chosen_particle, lidar_points)
        grid_coords = s.map.grid_cell_from_xz(lidar_in_world_coords[:, 0], lidar_in_world_coords[:, 2])
        # We will update the log-odds of the cells that are occupied
        s.map.log_odds[grid_coords[0], grid_coords[1]] += s.lidar_log_odds_occ - s.lidar_log_odds_free
        # We will update the log-odds of the cells that are free.
        free_cells = s.find_free_cells_from_obstacles(chosen_particle, lidar_in_world_coords)
        s.map.log_odds[free_cells[:, 0], free_cells[:, 1]] += s.lidar_log_odds_free
        # Clip log-odds
        s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)
        # Update binary map
        s.map.cells = s.map.log_odds >= s.map.log_odds_thresh
        # Convert to 0 and 1
        s.map.cells = s.map.cells.astype(np.int8)
            

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
