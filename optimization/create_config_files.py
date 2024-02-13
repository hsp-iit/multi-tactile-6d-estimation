import json
import logging
import argparse
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dir_name = os.path.abspath(os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--json_folder', dest='json_folder', help='absolute path to the json folder',
                        type=str, required=True)
    parser.add_argument('--weights_path', dest='weights', help='absolute path to the weights',
                        type=str, required=True)
    parser.add_argument('--working_directory_path', dest='working_directory', help='absolute path to the working directory',
                        type=str, required=True)
    args = parser.parse_args()

    # Create the image directory if necessary
    if not os.path.isdir(dir_name + "/results_directory_pose_estimation"):
        logger.info("Creating pose estimation directory..")
        os.mkdir(dir_name + "/results_directory_pose_estimation")

    results_directory = dir_name + "/results_directory_pose_estimation"

    # Objects considered
    with open(os.path.join(args.json_folder, 'objects.json')) as fo:
        list_objects = json.load(fo)

    # Weights of the autoencoder
    weights = args.weights

    # List of corresponding number of meshes for each object
    with open(os.path.join(args.json_folder, 'meshes.json')) as fm:
        list_meshes = json.load(fm)

    # Loop over the object to create the config files
    for index_obj, obj in enumerate(list_objects):

        if not os.path.isdir(results_directory + "/" + obj):
            os.mkdir(results_directory + "/" + obj)
        object_directory = results_directory + "/" + obj

        logger.info(f"Creating config files for {obj}..")

        for triplet in range(1,5,1):

            if not os.path.isdir(object_directory + "/triplet_" + str(triplet)):
                os.mkdir(object_directory + "/triplet_" + str(triplet))
            triplet_directory = object_directory + "/triplet_" + str(triplet)

            if not os.path.isdir(triplet_directory + "/config_directory"):
                os.mkdir(triplet_directory + "/config_directory")
            config_directory = triplet_directory + "/config_directory"

            if not os.path.isdir(triplet_directory + "/results_sensor"):
                os.mkdir(triplet_directory + "/results_sensor")

            if not os.path.isdir(triplet_directory + "/results_baseline"):
                os.mkdir(triplet_directory + "/results_baseline")

            f = open(config_directory + "/config_total_test.ini", "w")

            f.write("[Autoencoder] \n"
                    "enable_normalization = True \n"
                    "mean = 0.3839, 0.4071, 0.3787 \n"
                    "std = 0.1626, 0.1021, 0.0827 \n"
                    "prism_boolean = False \n"
                    "background = ../background.png \n"
                    "model = " + weights + " \n"
                    "batch_size = 4 \n"
                    "encoded_space = 128 \n"
                    "sub_background = None \n"
                    "[Files] \n"
                    "number_of_sensors = 3 \n"
                    "poses_images = " + args.working_directory + "/point_clouds/" + obj + "/poses_images.txt \n"
                    "poses_sensors = " + args.working_directory + "/final_triplets/" + obj + "/triplet_" + str(triplet) + "/sensors.off \n"
                    "depth = " + args.working_directory + "/phys_example/build/ \n"
                    "error = " + args.working_directory + "/errors/ \n"
                    "rotation = " + args.working_directory + "/rotations/ \n"
                    "angles_database = " + args.working_directory + "/angles_database.txt \n"
                    "point_cloud_file = " + args.working_directory + "/point_clouds/" + obj + "/poses_images.txt \n"
                    "image_sensor_1 = " + args.working_directory + "/final_triplets/" + obj + "/triplet_" + str(triplet) + "/Image_heatmap_0.png \n"
                    "image_sensor_2 = " + args.working_directory + "/final_triplets/" + obj + "/triplet_" + str(triplet) + "/Image_heatmap_1.png \n"
                    "image_sensor_3 = " + args.working_directory + "/final_triplets/" + obj + "/triplet_" + str(triplet) + "/Image_heatmap_2.png \n"
                    "images_point_cloud = " + args.working_directory + "/point_clouds/" + obj + "/images/ \n"
                    "[Parameters] \n"
                    "number_of_sensors = 3 \n"
                    "threshold = 0.04 \n"
                    "exponent = 8.5 \n"
                    "distance_threshold = 0.005 \n"
                    "rot_gain = 1.0 \n"
                    "pos_gain = 0.01 \n"
                    "iters = 700 \n"
                    "position_scalar = 0.2 \n"
                    "gt_x = 0.1 \n"
                    "gt_y = 0.0 \n"
                    "gt_z = 0.2 \n"
                    "loop = 15")
            f.close()

            f = open(config_directory + "/config_physx.ini", "w")
            f.write("MESH_PATH ../../models/" + obj + "/textured_ \n"
                    "NUMBER_OF_MESHES " + list_meshes[index_obj] + " \n"
                    "NUMBER_OF_SENSORS 3 \n"
                    "ROTATIONS_FILE_PATH ../../rotations/ \n"
                    "LANDMARKS_PATH ../../final_triplets/" + obj + "/triplet_" + str(triplet) + "/poses_sensors.txt")
            f.close()


if __name__ == '__main__':
    main()
