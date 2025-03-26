# Pratik Chaudhari (pratikac@seas.upenn.edu)
# Minku Kim (minkukim@seas.upenn.edu)

import click, tqdm, random

from slam import *

JUST_PLOT_MAP = False

def run_dynamics_step(src_dir, log_dir, idx, t0=0, draw_fig=False):
    """
    This function is for you to test your dynamics update step. It will create
    two figures after you run it. The first one is the robot location trajectory
    using odometry information obtained form the lidar. The second is the trajectory
    using the PF with a very small dynamics noise. The two figures should look similar.
    """
    slam = slam_t(Q=1e-8*np.eye(3))
    slam.read_data(src_dir, idx)

    # Trajectory using odometry (xz and yaw) in the lidar data
    d = slam.poses
    pose = np.column_stack([d[:,0,3], d[:,1,3], d[:,2,3]]) # X Y Z
    plt.figure(1)
    plt.clf()
    plt.title('Trajectory using onboard odometry')
    plt.plot(pose[:,0], pose[:,2])
    logging.info('> Saving odometry plot in '+os.path.join(log_dir, 'odometry_%s.jpg'%(idx)))
    plt.savefig(os.path.join(log_dir, 'odometry_%s.jpg'%(idx)))

    # dynamics propagation using particle filter
    # n: number of particles, w: weights, p: particles (3 dimensions, n particles)
    # S covariance of the xyth location
    # particles are initialized at the first xyth given by the lidar
    # for checking in this function
    n = 3
    w = np.ones(n)/float(n)
    p = np.zeros((3,n), dtype=np.float64)
    slam.init_particles(n,p,w)
    slam.p[:,0] = deepcopy(pose[0])

    print('> Running prediction')
    t0 = 0
    T = len(d)
    ps = deepcopy(slam.p)
    plt.figure(2)
    plt.clf()
    ax = plt.subplot(111)
    for t in tqdm.tqdm(range(t0+1,T)):
        slam.dynamics_step(t)
        ps = np.hstack((ps, slam.p))

        if draw_fig:
            ax.clear()
            ax.plot(slam.p[0], slam.p[0], '*r')
            plt.title('Particles %03d'%t)
            plt.draw()
            plt.pause(0.01)

    plt.plot(ps[0], ps[1], '*c')
    plt.title('Trajectory using PF')
    logging.info('> Saving plot in '+os.path.join(log_dir, 'dynamics_only_%s.jpg'%(idx)))
    plt.savefig(os.path.join(log_dir, 'dynamics_only_%s.jpg'%(idx)))

def run_observation_step(src_dir, log_dir, idx, is_online=False):
    """
    This function is for you to debug your observation update step
    It will create three particles np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]])
    * Note that the particle array has the shape 3 x num_particles so
    the first particle is at [x=0.2, y=0.4, z=0.1]
    This function will build the first map and update the 3 particles for one time step.
    After running this function, you should get that the weight of the second particle is the largest since it is the closest to the origin [0, 0, 0]
    """
    slam = slam_t(resolution=0.5)
    slam.read_data(src_dir, idx)

    # t=0 sets up the map using the yaw of the lidar, do not use yaw for
    # other timestep
    # initialize the particles at the location of the lidar so that we have some
    # occupied cells in the map to calculate the observation update in the next step
    t0 = 0
    d = slam.poses
    print(d[0])
    pose = np.column_stack([d[t0,0,3], d[t0,1,3], np.arctan2(-d[t0,2,0], d[t0,0,0])])
    logging.debug('> Initializing 1 particle at: {}'.format(pose))
    slam.init_particles(n=1,p=pose.reshape((3,1)),w=np.array([1]))

    slam.observation_step(t=0)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

    # reinitialize particles, this is the real test
    logging.info('\n')
    n = 3
    w = np.ones(n)/float(n)
    p = np.array([[2, 0.2, 3],[2, 0.4, 5],[2.7, 0.1, 4]])
    slam.init_particles(n, p, w)

    slam.observation_step(t=1)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

def run_slam(src_dir, log_dir, idx):
    """
    This function runs slam. We will initialize the slam just like the observation_step
    before taking dynamics and observation updates one by one. You should initialize
    the slam with n=50 particles, you will also have to change the dynamics noise to
    be something larger than the very small value we picked in run_dynamics_step function
    above.
    """
    slam = slam_t(resolution=0.5, Q=np.diag([1e-6,1e-6,1e-6]))
    slam.read_data(src_dir, idx)
    T = len(slam.lidar_files)
    Ts_poses = len(slam.poses)
    print('> Number of lidar scans: {}'.format(T))
    print('> Number of poses: {}'.format(Ts_poses))

    if not JUST_PLOT_MAP:
        # again initialize the map to enable calculation of the observation logp in
        # future steps, this time we want to be more careful and initialize with the
        # correct lidar scan
        #### TODO: XXXXXXXXXXXX
        # Initalize 1 particle at the first pose
        init_pose = slam.poses[0]
        init_particle = np.array([init_pose[0,3], init_pose[2,3], np.arctan2(init_pose[0, 2], init_pose[0,0])])
        slam.init_particles(n=1, p=init_particle.reshape((3,1)), w=np.array([1]))

        # initialize the map by running the observation step
        slam.observation_step(t=0)

        # initialize say n = 50 particles
        slam.init_particles(n=50, p=None, w=None)

        # run dynamics, observation and resampling steps for each timepoint
        largest_weight_particle = np.zeros((T, 3))
        for t in tqdm.tqdm(range(1, T)):
            slam.dynamics_step(t)
            slam.observation_step(t)

            # save the particle with the largest weight
            largest_weight_particle[t] = slam.p[:, np.argmax(slam.w)]

            slam.resample_particles()

        # Save the map into npy file
        
        # Plot the binary map
        binary_map = slam.map.cells.astype(np.uint8)
        logging.info('> Saving map in '+os.path.join(log_dir, 'map_%s.npy'%(idx)))
        np.save(os.path.join(log_dir, 'map_%s.npy'%(idx)), binary_map)

    else:
        # Just load the map from the npy file
        binary_map = np.load(os.path.join(log_dir, 'map_%s.npy'%(idx))).astype(np.uint8)

    binary_map = binary_map[np.min(np.where(np.sum(binary_map, axis=1) > 0)):np.max(np.where(np.sum(binary_map, axis=1) > 0)),
                            np.min(np.where(np.sum(binary_map, axis=0) > 0)):np.max(np.where(np.sum(binary_map, axis=0) > 0))]
    binary_map = (1 - binary_map) * 255

    # Interchange the rows and columns to plot the map in the same orientation as the lidar data
    binary_map = np.transpose(binary_map)
    # Reverse the rows to plot the map in the same orientation as the lidar data
    binary_map = binary_map[::-1, :]
    
    plt.figure(3)
    plt.clf()
    plt.imshow(binary_map, cmap='gray')
    plt.title('Binary map')
    logging.info('> Saving binary map in '+os.path.join(log_dir, 'map_%s.jpg'%(idx)))
    plt.savefig(os.path.join(log_dir, 'map_%s.jpg'%(idx)))

    # Plot trajectory of the largest weight particle and the actual trajectory
    # from the lidar data in the same plot with different colors.
    if not JUST_PLOT_MAP:
        plt.figure(4)
        plt.clf()
        d = slam.poses
        pose = np.column_stack([d[:,0,3], d[:,1,3], d[:,2,3]]) # X Y Z
        plt.plot(pose[:,0], pose[:,2], label='Actual trajectory')
        plt.plot(largest_weight_particle[:,0], largest_weight_particle[:,1], label='Estimated trajectory')
        plt.title('Trajectory')
        plt.legend()
        logging.info('> Saving trajectory plot in '+os.path.join(log_dir, 'trajectory_%s.jpg'%(idx)))
        plt.savefig(os.path.join(log_dir, 'trajectory_%s.jpg'%(idx)))


@click.command()
@click.option('--src_dir', default='./KITTI/', help='data directory', type=str)
@click.option('--log_dir', default='logs', help='directory to save logs', type=str)
@click.option('--idx', default='00', help='dataset number', type=str)
@click.option('--mode', default='slam',
              help='choices: dynamics OR observation OR slam', type=str)
def main(src_dir, log_dir, idx, mode):
    # Run python main.py --help to see how to provide command line arguments

    if not mode in ['slam', 'dynamics', 'observation']:
        raise ValueError('Unknown argument --mode %s'%mode)
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)

    if mode == 'dynamics':
        run_dynamics_step(src_dir, log_dir, idx)
        sys.exit(0)
    elif mode == 'observation':
        run_observation_step(src_dir, log_dir, idx)
        sys.exit(0)
    else:
        p = run_slam(src_dir, log_dir, idx)
        return p

if __name__=='__main__':
    main()
