import argparse
import configparser
from jaxlie import SO3
import numpy as np
import os

dir_name = os.path.abspath(os.path.dirname(__file__))

def save_points_as_off(points, name):
    with open(name, 'w') as out:
        out.write('OFF\r\n')
        out.write(str(points.shape[1]) + ' 0 0\r\n')
        for i in range(points.shape[1]):
            out.write(str(points[0, i]) + ' ' + str(points[1, i]) + ' ' + str(points[2, i]) + ' \r\n')

def save_pose(points, twist, name):
    R = SO3.exp(twist[3:6]).as_matrix()
    u, _, vh = np.linalg.svd(R, full_matrices = True)
    R = u @ vh
    points = R @ points + np.array([[twist[0]], [twist[1]], [twist[2]]])
    with open(name, 'w') as out:
        out.write('OFF\r\n')
        out.write(str(points.shape[1]) + ' 0 0\r\n')
        for i in range(points.shape[1]):
            out.write(str(points[0, i]) + ' ' + str(points[1, i]) + ' ' + str(points[2, i]) + ' \r\n')

def main():

    if not os.path.isdir(dir_name + "/remaining_poses"):
        print("Creating directory..")
        os.mkdir(dir_name + "/remaining_poses")

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file_path', dest='config_file_path', help='path to the config file',
                        type=str, required=True)

    args = parser.parse_args()

    config_file = args.config_file_path
    config  =configparser.ConfigParser()
    config.read(config_file)

    files = config['Files']
    data = np.loadtxt(files['poses_images'])
    points = data[:, :3].T

    arrays = np.load(os.path.join(dir_name + '/final_results.npz'))
    total_poses = arrays['total_poses']
    ordered_poses = arrays['ordered_poses']

    best_poses = 10

    for i in range(best_poses):

        index = int(ordered_poses[i])
        save_pose(points, total_poses[index, :], 'remaining_poses/best_' + str(i) + '.off')


if __name__ == '__main__':
    main()
