import configparser
import numpy as np
import sys


def main():
    """_
    Compute the total errors for each pose considering the interpenetration depth
    and the distance error between candidate points and sensors.
    """
    config_file = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    files = config['Files']

    # Number of different positions considered per rotation
    loop_range = int(files['loop'])

    # Initialize lists for poses and erorrs
    total_poses = []
    sum_errors = []
    position_errors = []
    pose_errors = []
    filtered_indexes = []
    depths_all = []
    errors_all = []

    # Load the arrays created by the optimization file
    arrays = np.load('arrays.npz')
    filtered = arrays['indexes']

    num_poses = int(filtered.shape[0]/loop_range)
    twists = arrays['twists']
    optimization_errors = arrays['errors']

    # Define ground truth position
    ground_truth_position = np.array([float(config['Constants']['gt_x']),
                                     float(config['Constants']['gt_y']),
                                     float(config['Constants']['gt_z'])])

    # We loop over the different initial positions the poses are initialized to
    for j in range(loop_range):
        depths = np.loadtxt(files['depth']+ "depths" + str(j) + ".txt")
        poses = twists[j*num_poses : (j+1)*num_poses, :]

        errors = optimization_errors[j*num_poses : (j+1)*num_poses]

        for i in range(depths.shape[0]):
            total_poses.append(poses[i])

            # Simply sum the two errors
            sum_errors.append(depths[i] + errors[i])
            errors_all.append(errors[i])
            depths_all.append(depths[i])

            # Subtract the ground truth positions, while the ground truth rotation is the identity
            position_errors.append(np.linalg.norm(poses[i,0:3] - ground_truth_position))
            pose_errors.append(np.linalg.norm(poses[i, 3:]))
            filtered_indexes.append(filtered[i])

    # Sort the errors
    ordered = np.argsort(np.array(sum_errors))

    np.savez('final_results.npz', total_poses=np.array(total_poses), 
             sum_errors=np.array(sum_errors), position_errors=np.array(position_errors),
             pose_errors=np.array(pose_errors), ordered_poses=np.array(ordered), 
             all_depths=np.array(depths_all), all_errors=np.array(errors_all))

if __name__ == '__main__':

    main()
