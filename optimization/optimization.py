import configparser
import copy
import jax.numpy as jnp
import numpy as np
import pyquaternion as Quaternion
import random
import sys
import time

from tactile_based_selector import TactileBasedSelector
from itertools import combinations
from jax import jit, vmap, grad
from jaxlie import SO3
from scipy.stats import special_ortho_group
from tqdm import tqdm

np.random.seed(0)
random.seed(0)

def generate_indexes(indexes_list:list) -> np.array:
    """
    Generate a numpy array with the indexes of the cartesian product between numpy arrays.
    
    Args:
        indexes_list (list): list of arrays .containing the indexes of the points of the point cloud selected for every sensor

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
            out.write(str(points[0, i]) + ' ' + str(points[1, i]) + ' ' + str(points[2, i]) + ' \r\n')


def save_pose(points: np.array, twist: np.array, name: str) -> None:
    """
    Transform a point cloud with a given twist and then save the point cloud as .off.

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

    # Initialize the config_file
    config_file = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)

    # Initialize the TactileBasedSelector
    tactile_based_selector = TactileBasedSelector(config_file)
    tactile_based_selector.calculate_indexes()

    # Load the points_coordinate
    data = np.loadtxt(config['Files']['poses_images'])
    points = data[:, :3].T

    # Save the ground truth pose. In our setup was fixed
    save_pose(points, np.array([0.1, 0.0, 0.2, 0.0, 0.0, 0.0]), './real_pose.off')

    # Load the pose of the sensors.
    poses_sensors = np.loadtxt(config['Files']['poses_sensors'], skiprows = 2)

    # Set landmarks (i.e. sensors positions)
    landmarks = np.zeros(shape = (3, tactile_based_selector.number_of_sensors))
    for i in range(tactile_based_selector.number_of_sensors):
        landmarks[:, i] = poses_sensors[i,:3]

    # Save the position of contact of the sensors for visualization purpose.
    save_points_as_off(landmarks, './landmarks.off')


    # Distance-based selection method
    def eval_triplet(item: jnp.array, points: np.array, landmarks: np.array, comb: tuple, threshold: float=0.005)-> jnp.bool:
        """
        Evaluate the triplet based on the proprioception and return True if the average error is under a certain threshold.

        Args:
            item (jnp.array): the triplet under study
            points (np.array): array of points
            landmarks (np.array): positions of the sensors
            comb (tuple): tuple with combinations of the n sensors
            threshold (float, optional): threshold not to be passed by the triplet to be accepted. Defaults to 0.005.

        Returns:
            jnp.bool
        """
        
        list_errors = []

        for combination in comb:
            distance_landmarks = jnp.linalg.norm(landmarks[:, combination[0]] - landmarks[:, combination[1]])
            distance_points = jnp.linalg.norm(points[:, item[combination[0]]] - points[:, item[combination[1]]])
            list_errors.append(abs(distance_points - distance_landmarks))
        error = sum(list_errors)/len(list_errors)

        return jnp.bool_((jnp.heaviside(jnp.min(jnp.array([threshold - error, 0.0])), 1.0)))

    # Distance-based selection method jax implementation
    def eval_triplets(indexes: jnp.array, points: np.array, landmarks: np.array, comb: tuple, threshold: float=0.005)-> jnp.array:
        return vmap(eval_triplet, in_axes = (1, None, None, None, None))(indexes, points, landmarks, comb, threshold)


    eval_triplets = jit(eval_triplets)

    # Define the starting distance threshold
    distance_threshold = 0.005

    # Initialize array to store the filtered tuples
    filter_total = np.empty((tactile_based_selector.number_of_sensors, 0), dtype=np.int16)

    # Store the possible combinatory combinations between sensors
    comb_list = list(combinations(np.arange(tactile_based_selector.number_of_sensors), 2))

    # Check whether the code can be simplified in case of one or two sensors
    if tactile_based_selector.number_of_sensors == 1:
        indexes = np.array([tactile_based_selector.indexes_list[0]], dtype=np.int16)

    elif tactile_based_selector.number_of_sensors == 2:
        indexes = generate_indexes(tactile_based_selector.indexes_list)

    else:

        # Control logic to handle the limited gpu memory
        points_total = 1
        loop_bool = False

        for sens in range(tactile_based_selector.number_of_sensors - 1):
            points_total *= tactile_based_selector.indexes_list[sens+1].shape[0]

        step_max = (10 ** 10)/ points_total

        # Check if more than one step of distance-based selection is needed
        if step_max < tactile_based_selector.indexes_list[0].shape[0]:
            step = step_max
            loop_range, remaining = divmod(tactile_based_selector.indexes_list[0].shape[0], remaining)
            loop_bool = True

        else:
            loop_bool = False

        if loop_bool:
            # Loop to filter all the triplets
            for i in range(loop_range):
                indexes = generate_indexes(list(np.array([tactile_based_selector.indexes_list[0][i * step : (i+1) * step]])) + tactile_based_selector.indexes_list[1:])
                indexes = jnp.array(indexes)

                filtered = eval_triplets(indexes, points, landmarks, tactile_based_selector.number_of_sensors, comb_list, distance_threshold)

                filtered_indexes = indexes[:, filtered]

                filter_total = np.append(filter_total, np.array(filtered_indexes), 1)

                filtered_indexes = None
                indexes = None
                filtered = None
            if remaining != 0:
                indexes = generate_indexes(list(np.array([tactile_based_selector.indexes_list[0][loop_range* step :]])) + tactile_based_selector.indexes_list[1:])
                indexes = jnp.array(indexes)

                filtered = eval_triplets(indexes, points, landmarks, tactile_based_selector.number_of_sensors, comb_list, distance_threshold)

                filtered_indexes = indexes[:, filtered]

                filter_total = np.append(filter_total, np.array(filtered_indexes), 1)

                filtered_indexes = None
                indexes = None
                filtered = None
        
        # Generate all the indexes all at once
        else:
            indexes = generate_indexes(tactile_based_selector.indexes_list)
            indexes = jnp.array(indexes)

            filtered = eval_triplets(indexes, points, landmarks, tactile_based_selector.number_of_sensors, comb_list, distance_threshold)

            filtered_indexes = indexes[:, filtered]
            filter_total = np.append(filter_total, np.array(filtered_indexes), 1)
            filtered_indexes = None
            indexes = None
            filtered = None

    indexes = filter_total
    points = jnp.array(points)
    landmarks = jnp.array(landmarks)

    boolean_loop = False
    boolean_triplets = True

    if tactile_based_selector.number_of_sensors != 0:

        # Check whether the number of triplets to be studied is appropriate
        while boolean_triplets:

            filter = eval_triplets(indexes, points, landmarks, tactile_based_selector.number_of_sensors, comb_list, distance_threshold)


            filtered_indexes = indexes[:, filter]
            print(filtered_indexes.shape)

            if filtered_indexes.shape[1] < 5000 and filtered_indexes.shape[1] >0:
                boolean_triplets = False

            elif filtered_indexes.shape[1] > 5000 and boolean_loop:
                boolean_triplets = False

            elif filtered_indexes.shape[1] == 0:
                distance_threshold += 0.0001
                boolean_loop = True
                filter = None
            else:
                distance_threshold -= 0.0005
                filtered_indexes = None
    else:
        filtered_indexes = indexes

    filtered_indexes = jnp.array(filtered_indexes)
    print("Remaining indexes shape: ")
    print(filtered_indexes.shape)

    # Loss
    def eval_item(item, points, landmarks, twist):
        T = SO3.exp(twist[3:6]).as_matrix()
        error = 0
        for i in range(landmarks.shape[1]):
            error += jnp.linalg.norm((T @ points[:, item[i]] + twist[0:3]) - landmarks[:, i])

        return error / landmarks.shape[1]

    def eval_items(twist, filtered_index):
        return eval_item(filtered_index, points, landmarks, twist)

    def eval_items_final(twists):
        return vmap(eval_items, in_axes=(1,1))(twists, filtered_indexes)

    g = grad(eval_items)

    def grads(twists):
        return vmap(g, in_axes=(1,1))(twists, filtered_indexes)

    grads = jit(grads)

    # Define the iteration of the optimization process.
    # 700 represent an heuristic solution
    iters = 700

    # Initialize the centroid
    x0 = np.average(landmarks[0, :])
    y0 = np.average(landmarks[1, :])
    z0 = np.average(landmarks[2, :])

    # Save the time to understand how much time we need to generate the random poses
    start_time = time.time()
    q = Quaternion.Quaternion(axis=[1, 0, 0], angle=0)

    pose = np.concatenate((np.array([x0, y0, z0]), q.axis * q.angle), axis=0)


    twists = np.repeat(np.expand_dims(pose, 1), filtered_indexes.shape[1], axis=1)

    # Generate as many random rotations as the number of triplets under consideration
    for i in range(filtered_indexes.shape[1]):

        q = Quaternion.Quaternion(matrix = special_ortho_group.rvs(3))

        twists[0:3, i] = np.array([x0, y0, z0])
        twists[3:6, i] = q.axis * q.angle

    print("Time to generate poses: ")
    print(time.time() - start_time)

    # Consider all around positions for the poses since the precision of the GD can depend on the starting pose.
    positions = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 1], [-1, -1, -1], [1, -1, 1], [1, -1, -1], [1, 1, -1], [-1, -1, 1],[-1, 1, 1], [-1, 1, -1]])

    # Multiply the positions of a quantity which is greater than the bigger half dimension of the object considered.
    # Chose 0.2, which is greater than the half of biggest object considered
    position_multiplier = 0.2

    for position in range(positions.shape[0]):

        # Move the poses on the position set
        twists_copy = copy.deepcopy(twists)
        twists_copy[0] += positions[position][0] * position_multiplier
        twists_copy[1] += positions[position][1] * position_multiplier
        twists_copy[2] += positions[position][2] * position_multiplier
        twists_copy = jnp.array(twists_copy)

        # Gradient descent tuning
        K = np.eye(6)
        K[0:3, 0:3] *= 0.01
        K[3:6, 3:6] *= 1.0
        K = jnp.array(K)

        for i in tqdm(range(iters)):

            variable = grads(twists_copy).T

            # Optimize
            twists_copy = twists_copy - K @ variable

        errors = eval_items_final(twists_copy)

        errors = np.array(errors)

        f = open("rotations_best"+str(position)+".txt", "w")
        f.close()
        f = open("errors_best" + str(position) + ".txt", "w")
        f.close()
        f = open("filtered_indexes" + str(position) + ".txt", "w")
        f.close()


        np.savetxt("rotations_best"+str(position)+".txt", twists_copy.T)
        np.savetxt("errors_best" + str(position) + ".txt", errors)
        np.savetxt("filtered_indexes" + str(position) + ".txt", filtered_indexes.T)


if __name__ == '__main__':
    main()
