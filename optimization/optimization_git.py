import argparse
import configparser
import copy
import jax.numpy as jnp
import logging
import numpy as np
import pyquaternion as Quaternion
import random
import time

from tactile_based_selector_test import TactileBasedSelector
from itertools import combinations
from jax import jit, vmap, grad
from jaxlie import SO3
from scipy.stats import special_ortho_group
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(0)
random.seed(0)

def generate_indexes(indexes_list:list) -> np.array:
    """
    Generate a numpy array with the indexes of the cartesian product between numpy arrays.

    Args:
        indexes_list (list): list of arrays .containing the indexes of the points of the point cloud
        selected for every sensor

    Returns:
        np.array

    """

    meshgrid = np.meshgrid(*indexes_list, indexing='ij')
    coordinate_grid = np.array((meshgrid), dtype=np.int16)

    total_number = np.prod([index.shape[0] for index in indexes_list])

    return np.reshape(coordinate_grid, (len(indexes_list), total_number))


def save_points_as_off(points: np.array, name: str) -> None:
    """
    Save a point cloud as off given the array of the points and the name.

    Args:
        points (np.array): array of points
        name (str): name of the file

    """

    with open(name, 'w') as out:
        out.write('OFF\r\n')
        out.write(str(points.shape[1]) + ' 0 0\r\n')
        for i in range(points.shape[1]):
            out.write(str(points[0, i]) + ' ' + str(points[1, i]) + ' ' +
                      str(points[2, i]) + ' \r\n')

def save_pose(points: np.array, twist: np.array, name: str) -> None:
    """
    Transform a point cloud with a given twist and then save the point cloud
    as .off.

    Args:
        points (np.array): array of points
        twist (np.array): twist to be applied to the point cloud
        name (str): name of the file

    """

    R = SO3.exp(twist[3:6]).as_matrix()
    u, _, vh = np.linalg.svd(R, full_matrices = True)
    R = u @ vh
    points = R @ points + np.array([[twist[0]], [twist[1]], [twist[2]]])

    save_points_as_off(points, name)


def main():

    # Initialize the parser and parse the inputs
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file_path', dest='config_file_path', help='path to the config file',
                        type=str, required=True)

    parser.add_argument('--enable_selection', dest='enable_selection',
                        help='enable tactile selection', type=bool, default=False)

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file_path)

    # Initialize the TactileBasedSelector
    tactile_based_selector = TactileBasedSelector(args.config_file_path)

    # Check whether to select or not the points based on the contact
    # information
    tactile_selection = args.enable_selection
    tactile_based_selector.calculate_indexes(tactile_selection)
    logger.info(f'Tactile based selection: {tactile_selection}')

    # Load the points_coordinate
    data = np.loadtxt(config['Files']['poses_images'])
    points = data[:, :3].T

    # Load the exponent to keep into account hardware limit.
    # For example, for a 24GB GPU it is convenient to use 8.5
    exponent = float(config['Parameters']['exponent'])

    # Define the starting distance threshold
    distance_threshold = float(config['Parameters']['distance_threshold'])

    # Load the gain parameters of the optimizer
    rot_gain = float(config['Parameters']['rot_gain'])
    pos_gain = float(config['Parameters']['pos_gain'])

    # Load the pose of the sensors.
    poses_sensors = np.loadtxt(config['Files']['poses_sensors'], skiprows = 2)

    # Set landmarks (i.e. sensors positions)
    landmarks = np.zeros(shape = (3, tactile_based_selector.number_of_sensors))
    for i in range(tactile_based_selector.number_of_sensors):
        landmarks[:, i] = poses_sensors[i,:3]

    # Distance-based selection method
    def eval_tuple(ntuple: jnp.array, points: np.array, landmarks: np.array,
                   comb: tuple, threshold: float)-> jnp.bool_:
        """
        Evaluate the n-tuple of sensors depending on the actual position of the sensors given by the
        proprioception.
        Return true if the n-tuple is valid, i.e. the distance between the sensors in the n-tuple is
        compatible with that of the sensors as given by the proprioception, false otherwise.

        Args:
            ntuple (jnp.array): the n-tuple containing the indexes to be used to access the array of
                points and obtain the corresponding 3D Cartesian position of the sensor
            points (np.array): array of all points, each to be considered as the Cartesian position
                of one sensor
            landmarks (np.array): positions of the sensors from the proprioception
            comb (tuple): tuple with all possible connections between N sensors. For example: for
                three sensors ((0, 1), (0, 2), (1, 2))
            threshold (float, optional): it decides if the n-tuple is accepted or not by comparing
                the difference between the mutual landmarks distance and the mutual points distance
                averaged across all combinations in comb

        Returns:
            jnp.bool
        """

        list_errors = []

        for combination in comb:
            distance_landmarks = jnp.linalg.norm(landmarks[:, combination[0]] -
                                                 landmarks[:, combination[1]])
            distance_points = jnp.linalg.norm(points[:, ntuple[combination[0]]] -
                                              points[:, ntuple[combination[1]]])
            list_errors.append(abs(distance_points - distance_landmarks))
        error = sum(list_errors)/len(list_errors)

        return jnp.bool_((jnp.heaviside(jnp.min(jnp.array([threshold - error, 0.0])), 1.0)))

    # Distance-based selection method jax implementation
    def eval_ntuples(indexes: jnp.array, points: np.array, landmarks: np.array, comb: tuple,
                     threshold: float)-> jnp.array:
        return vmap(eval_tuple, in_axes = (1, None, None, None, None))(indexes, points, landmarks,
                                                                       comb, threshold)

    eval_ntuples = jit(eval_ntuples)

    # Initialize array to store the filtered tuples
    filter_total = np.empty((tactile_based_selector.number_of_sensors, 0), dtype=np.int16)

    # Evaluate all the possible combinations of N sensors
    # For example: for three sensors ((0, 1), (0, 2), (1, 2))
    comb_list = list(combinations(np.arange(tactile_based_selector.number_of_sensors), 2))

    # Check whether the code can be simplified in case of one or two sensors
    if tactile_based_selector.number_of_sensors == 1:
        indexes = np.array([tactile_based_selector.indexes_list[0]], dtype=np.int16)

    elif tactile_based_selector.number_of_sensors == 2:
        indexes = generate_indexes(tactile_based_selector.indexes_list)

    else:

        # Control logic to handle the limited gpu memory
        points_total = 1
        loop_bool = True

        for sens in range(tactile_based_selector.number_of_sensors):
            points_total *= tactile_based_selector.indexes_list[sens].shape[0]
        total_val = (10 ** exponent)

        if total_val/points_total >=1:
            loop_bool = False
        else:
            index_list = 0
            last_val = total_val
            while True:
                new_val = (last_val + index_list - tactile_based_selector.number_of_sensors)/(
                               tactile_based_selector.indexes_list[index_list].shape[0])
                if new_val < 1:
                    break
                index_list += 1
                last_val = new_val

            remaining = int(last_val/(tactile_based_selector.number_of_sensors - index_list))
            lists_of_chuncks = [ [] for _ in range(tactile_based_selector.number_of_sensors)]


            lists_indexes_of_chuncks = copy.deepcopy(lists_of_chuncks)

            for sens in reversed(range(tactile_based_selector.number_of_sensors)):
                if sens>= tactile_based_selector.number_of_sensors - index_list:
                    lists_of_chuncks[sens] = [0, tactile_based_selector.indexes_list[sens].shape[0]]

                else:
                    number_of_chuncks = tactile_based_selector.indexes_list[sens].shape[0] / (
                        remaining)

                    lists_of_chuncks[sens] = list(map( lambda l : remaining*l, list(range(int(
                        number_of_chuncks)))))+ [tactile_based_selector.indexes_list[sens].shape[0]]

                lists_indexes_of_chuncks[sens] = np.arange(len(lists_of_chuncks[sens])-1)

            indexes_to_filter = generate_indexes(lists_indexes_of_chuncks).astype('int')

        if loop_bool:
            # Loop to filter all the ntuples
            for i in range(indexes_to_filter.shape[1]):
                list_to_parse = []
                for sens in range(tactile_based_selector.number_of_sensors):

                    list_to_parse.append(tactile_based_selector.indexes_list[sens][
                        lists_of_chuncks[sens][indexes_to_filter[sens,i]]:
                        lists_of_chuncks[sens][indexes_to_filter[sens,i]+1]])

                #if np.all(list_to_parse[0] == tactile_based_selector.indexes_list[0]):

                indexes = generate_indexes(list_to_parse)

                indexes = jnp.array(indexes)

                filtered = eval_ntuples(indexes, points, landmarks, comb_list, distance_threshold)

                filtered_indexes = indexes[:, filtered]

                filter_total = np.append(filter_total, np.array(filtered_indexes), 1)

                filtered_indexes = None
                indexes = None
                filtered = None




        # Generate all the indexes all at once
        else:

            indexes = generate_indexes(tactile_based_selector.indexes_list)

            indexes = jnp.array(indexes)

            filtered = eval_ntuples(indexes, points, landmarks, comb_list, distance_threshold)

            filtered_indexes = indexes[:, filtered]
            filter_total = np.append(filter_total, np.array(filtered_indexes), 1)

        logger.info(f'Tuples selected: {filter_total.shape}')
        indexes = filter_total

        points = jnp.array(points)
        landmarks = jnp.array(landmarks)

    boolean_loop = False
    boolean_ntuples = True

    if tactile_based_selector.number_of_sensors != 0:

        # Check whether the number of ntuples to be studied is appropriate
        while boolean_ntuples:

            filter = eval_ntuples(indexes, points, landmarks, comb_list, distance_threshold)

            filtered_indexes = indexes[:, filter]

            # Print the actual number of tuples
            logger.info(f'Number of tuples after round of distance-based selection: {filtered_indexes.shape}')

            # Check if we have a sufficient number of ntuples to consider yet not too high
            if filtered_indexes.shape[1] < 5000 and filtered_indexes.shape[1] >0:
                boolean_ntuples = False

            # Check whether we already increased the distance threshold even if the number of
            # ntuples is high
            elif filtered_indexes.shape[1] > 5000 and boolean_loop:
                boolean_ntuples = False

            # Check whether we set a distance threshold too low
            elif filtered_indexes.shape[1] == 0:
                distance_threshold += 0.0001
                boolean_loop = True
                filter = None
            # Decrease the distance threshold
            else:
                distance_threshold -= 0.0005
                filtered_indexes = None
    else:
        filtered_indexes = indexes

    # Print the final number of tuples
    logger.info(f"Remaining number of tuples: {filtered_indexes.shape}")

    num_poses = filtered_indexes.shape[1]

    # Define the iteration of the optimization process.
    iters = int(config['Parameters']['iters'])

    # Initialize the centroid
    x0 = np.average(landmarks[0, :])
    y0 = np.average(landmarks[1, :])
    z0 = np.average(landmarks[2, :])

    # Time benchmarking
    start_time = time.time()
    initial_quaternion = Quaternion.Quaternion(axis=[1, 0, 0], angle=0)

    pose = np.concatenate((np.array([x0, y0, z0]), initial_quaternion.axis * initial_quaternion.angle), axis=0)

    twists = np.repeat(np.expand_dims(pose, 1), num_poses, axis=1)

    # Generate as many random rotations as the number of ntuples under consideration
    for i in range(num_poses):

        initial_quaternion = Quaternion.Quaternion(matrix = special_ortho_group.rvs(3))

        twists[0:3, i] = np.array([x0, y0, z0])
        twists[3:6, i] = initial_quaternion.axis * initial_quaternion.angle

    logger.info(f"Time required to initialize the poses: {time.time() - start_time}")

    # For each given orientation, consider alternatives for the initial position around the centroid.
    directions = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
                           [0, 0, -1], [1, 1, 1], [-1, -1, -1], [1, -1, 1], [1, -1, -1], [1, 1, -1],
                           [-1, -1, 1],[-1, 1, 1], [-1, 1, -1]])

    # Use the following scaler to shift the positions along the above define directions
    position_scalar = float(config['Parameters']['position_scalar'])

    # Initialize the total arrays
    twists_total = np.empty((6,0))
    filtered_indexes_total = np.empty((3,0))

    for position in range(directions.shape[0]):
        twists_copy = copy.deepcopy(twists)
        twists_copy[0] += directions[position][0] * position_scalar
        twists_copy[1] += directions[position][1] * position_scalar
        twists_copy[2] += directions[position][2] * position_scalar

        filtered_indexes_total = np.concatenate((filtered_indexes_total,filtered_indexes), axis=1)
        twists_total = np.concatenate((twists_total, twists_copy), axis=1)

    twists_total = jnp.array(twists_total)
    filtered_indexes_total = jnp.array(filtered_indexes_total.astype('int32'))


    # We calculate the loss based on the distance between the candidate ntuple
    # transformed with the actual twist and the position of the sensors
    def eval_item(item, points, landmarks, twist):
        T = SO3.exp(twist[3:6]).as_matrix()
        error = 0
        for i in range(landmarks.shape[1]):
            error += jnp.linalg.norm((T @ points[:, item[i]] + twist[0:3]) - landmarks[:, i])

        return error / landmarks.shape[1]

    def eval_items(twist, filtered_index):
        return eval_item(filtered_index, points, landmarks, twist)

    def eval_items_final(twists):
        return vmap(eval_items, in_axes=(1,1))(twists, filtered_indexes_total)

    g = grad(eval_items)

    # The in_axes option selects on which axis we should parallelize the vectors
    def grads(twists):
        return vmap(g, in_axes=(1,1))(twists, filtered_indexes_total)

    grads = jit(grads)


    # Gradient descent tuning
    K = np.eye(6)
    K[0:3, 0:3] *= pos_gain
    K[3:6, 3:6] *= rot_gain
    K = jnp.array(K)

    for i in tqdm(range(iters)):

        variable = grads(twists_total).T

        # Optimize
        twists_total = twists_total - K @ variable

    errors_total = eval_items_final(twists_total)

    for position in range(directions.shape[0]):
        np.savetxt("rotations_best"+str(position)+".txt", twists_total.T[position*num_poses:
                                                                         (position+1)*num_poses,:])

    # Save the the final poses, the associated errors and the associated indexes of the point cloud
    np.savez('arrays.npz', twists=twists_total.T, errors=errors_total,
             indexes=filtered_indexes_total.T)

    # Save the position of contact of the sensors for visualization purpose.
    save_points_as_off(landmarks, './landmarks.off')

if __name__ == '__main__':
    main()
