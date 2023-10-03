import configparser
import numpy as np
import sys


def main():

    config_file = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)
    files = config['Files']

    # Number of different positions considered per rotation
    loop_range = int(files['loop'])

    # Initialize lists
    total_rotations = []
    sum_errors = []
    position_errors = []
    rotation_errors = []
    filtered_indexes = []
    depths_all = []
    errors_all = []

    for j in range(loop_range):
        rotations = np.loadtxt(files['rotation'] + "rotations_best" + str(j) + ".txt")
        depths = np.loadtxt(files['depth']+ "depths" + str(j) + ".txt")
        errors = np.loadtxt(files['error'] + "errors_best" + str(j) + ".txt")
        filtered = np.loadtxt(files['index'] + 'filtered_indexes' + str(j) + '.txt')

        for i in range(depths.shape[0]):
            total_rotations.append(rotations[i])

            # Simply sum the two errors
            sum_errors.append(depths[i] + errors[i])
            errors_all.append(errors[i])
            depths_all.append(depths[i])

            # Subtract the ground truth positions, while the ground truth rotation is the identity
            position_errors.append(np.linalg.norm(rotations[i,0:3] - np.array([0.1, 0.0, 0.2])))
            rotation_errors.append(np.linalg.norm(rotations[i, 3:]))
            filtered_indexes.append(filtered[i])
    
    # Cast the lists to numpy array
    filtered_indexes = np.array(filtered_indexes)
    total_rotations = np.array(total_rotations)
    sum_errors = np.array(sum_errors)
    position_errors = np.array(position_errors)
    rotation_errors = np.array(rotation_errors)
    depths_all = np.array(depths_all)
    errors_all = np.array(errors_all)

    # Sort the errors
    ordered = np.argsort(sum_errors)

    
    np.savetxt("all_rotations.txt", total_rotations)
    np.savetxt("sum_errors.txt", sum_errors)
    np.savetxt("position_errors.txt", position_errors)
    np.savetxt("rotation_errors.txt", rotation_errors)
    np.savetxt("ordered_poses.txt", ordered)
    np.savetxt("depths_all.txt", depths_all)
    np.savetxt("errors_all.txt", errors_all)
    np.savetxt("total_filtered_indexes.txt", filtered_indexes)

if __name__ == '__main__':

    main()
