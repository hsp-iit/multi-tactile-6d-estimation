import argparse
import numpy as np
import pyquaternion as pyq
from jaxlie import SO3
import os
import copy
import json
from bop_pose_error import adi as bop_adi
from bop_pose_error import VOCap

dir_name = os.path.dirname(__file__)
class ADIEvaluator():

    def __init__(self, ycbv_models_path, object_names):

        self.clouds = { object_name : self.load_point_cloud(os.path.join(ycbv_models_path, 'models', object_name, 'points.xyz')) for object_name in object_names}


    def load_point_cloud(self, file_name):
        """Load the xyz point cloud provided within the YCB Video Models folder."""

        data = []

        with open(file_name, newline = '') as file_data:
            for row in file_data:
                data.append([float(num_string.rstrip()) for num_string in row.rstrip().split(sep = ' ') if num_string != ''])
        return np.array(data)


    def adi_auc(self, object_name, reference, signal):
        """Evaluate distances and AUC for the Average Distance Indistinguishable (ADI) metric.

           object_name is the object name, e.g., '002_master_chef_can'
           reference is a [N, 7] matrix containing N ground truth poses each expressed as [x, y, z, axis_x, axis_y, axis_z, angle]
           signal is a [N, 7] matrix containing N ground truth poses each expressed as [x, y, z, axis_x, axis_y, axis_z, angle]

           Return
           - the ADI distances associated to all poses
           - the Area Under the Curve (AUC) of the ADI distances
        """

        distances = None
        sig = signal
        ref = reference

        dists = []
        for j in range(ref.shape[0]):
            gt_t = reference[j, 0 : 3]
            gt_R = pyq.Quaternion(axis = reference[j, 3 : 6], angle = reference[j, 6]).rotation_matrix

            estimate_t = signal[j, 0 : 3]
            estimate_R = pyq.Quaternion(axis = signal[j, 3 : 6], angle = signal[j, 6]).rotation_matrix

            dists.append(bop_adi(estimate_R, estimate_t, gt_R, gt_t, self.clouds[object_name]))

        distances = np.array(dists)

        distances_copy = copy.deepcopy(distances)
        threshold = 0.02
        inf_indexes = np.where(distances > threshold)[0]
        distances[inf_indexes] = np.inf
        sorted_distances = np.sort(distances)
        n = len(sorted_distances)
        accuracy = np.cumsum(np.ones((n, ), np.float32)) / n

        return distances_copy, VOCap(sorted_distances, accuracy) * 100.0



def calculate_auc(object_string, poses, num_poses, ycb_video_models_path, ground_truth_position, object_names):
    evaluator = ADIEvaluator(ycb_video_models_path, object_names)
    gt = np.concatenate((ground_truth_position, np.array([1, 0, 0, 0])))
    gt = np.repeat(np.expand_dims(gt, 0), num_poses, axis=0)

    dists, auc = evaluator.adi_auc(object_string, gt, poses)

    return auc

def calculate_auc_rot(object_string, poses, num_poses, ycb_video_models_path, ground_truth_position, object_names, bool_val = False):
    evaluator = ADIEvaluator(ycb_video_models_path, object_names)
    poses[:,:3] = ground_truth_position
    gt = np.concatenate((ground_truth_position, np.array([1, 0, 0, 0])))
    gt = np.repeat(np.expand_dims(gt, 0), num_poses, axis=0)

    dists, auc = evaluator.adi_auc(object_string, gt, poses)

    return auc



def create_table_all_objects(dict_objects, results_directory, list_keys):

    list_objects_tex = []
    for obj in list_keys:
        list_words = obj.rsplit('_')
        create_string = [x + r'{\_}'for x in list_words[:-1]]
        create_string.append(list_words[-1])
        create_string.append(' & ')
        list_objects_tex.append(''.join(create_string))

    with open(results_directory + "/table_all_objects.tex", 'w') as tf:
        tf.write(r'\begin{table*}')
        tf.write('\n')
        tf.write(r'\tiny')
        tf.write('\n')
        tf.write(r'\scriptsize')
        tf.write('\n')
        tf.write(r'\centering')
        tf.write('\n')
        tf.write(r'\caption{Table 1: Comparison between the baseline and the proposed method executed on simulated data using 10 objects.}')
        tf.write('\n')
        tf.write(r'\begin{tabular}{| l | c | c | c | c | c | c | c | c | c | c |}')
        tf.write('\n')
        tf.write(r'\hline')
        tf.write('\n')
        tf.write(r'metric & \multicolumn {2}{c|}{Positional error (cm))} & \multicolumn {2}{c|}{ADI-AUC (\%)} & \multicolumn {2}{c|}{ADI-R-AUC (\%)} & \multicolumn{2}{c|}{Contacts}\\')
        tf.write('\n')
        tf.write(r'\hline')
        tf.write('\n')
        tf.write(r'Method & Baseline & Ours & Baseline & Ours & Baseline & Ours & Baseline & Ours  \\')
        tf.write('\n')
        tf.write(r'\hline \\')
        tf.write('\n')
        tf.write(r'Evaluated poses & \#1, \#1-5 & \#1, \#1-5 & \#1, \#1-5 & \#1, \#1-5 & \#1, \#1-5 & \#1, \#1-5 & \#1, \#1-5 & \#1, \#1-5  \\')
        tf.write('\n')
        tf.write(r'\hline')
        tf.write('\n')
        for obj in range(len(list_objects_tex)):
            tf.write(list_objects_tex[obj]
                     + str('%.2f'%(dict_objects[list_keys[obj]][4]['Baseline']["Mean"][1])) + ', ' + str('%.2f'%(dict_objects[list_keys[obj]][4]['Baseline']["Mean"][4])) + ' & '
                     + str('%.2f'%(dict_objects[list_keys[obj]][4]['Sensor']["Mean"][1])) + ', ' + str('%.2f'%(dict_objects[list_keys[obj]][4]['Sensor']["Mean"][4])) + ' & '
                     + str('%.2f'%(dict_objects[list_keys[obj]][4]['Baseline']["Mean"][12])) + ', ' + str('%.2f'%(dict_objects[list_keys[obj]][4]['Baseline']["Mean"][13])) + ' & '
                     + str('%.2f'%(dict_objects[list_keys[obj]][4]['Sensor']["Mean"][12])) + ', ' + str('%.2f'%(dict_objects[list_keys[obj]][4]['Sensor']["Mean"][13])) + ' & '
                     + str('%.2f'%(dict_objects[list_keys[obj]][4]['Baseline']["Mean"][14])) + ', ' + str('%.2f'%(dict_objects[list_keys[obj]][4]['Baseline']["Mean"][15])) + ' & '
                     + str('%.2f'%(dict_objects[list_keys[obj]][4]['Sensor']["Mean"][14])) + ', ' + str('%.2f'%(dict_objects[list_keys[obj]][4]['Sensor']["Mean"][15])) + ' & '

                     + str('%.2f'%(dict_objects[list_keys[obj]][4]['Baseline']["Mean"][9])) + ', ' + str('%.2f'%(dict_objects[list_keys[obj]][4]['Baseline']["Mean"][10])) + ' & '
                     + str('%.2f'%(dict_objects[list_keys[obj]][4]['Sensor']["Mean"][9])) + ', ' + str('%.2f'%(dict_objects[list_keys[obj]][4]['Sensor']["Mean"][10])))
            tf.write(r'\\')
            tf.write('\n')

        tf.write(r'\hline')
        tf.write('\n')
        tf.write(r'Mean & '
                 + str('%.2f'%(dict_objects["total_objects"]['Baseline']["Mean"][1])) + ', ' + str('%.2f'%(dict_objects["total_objects"]['Baseline']["Mean"][4])) + ' & '
                 + str('%.2f'%(dict_objects["total_objects"]['Sensor']["Mean"][1])) + ', ' + str('%.2f'%(dict_objects["total_objects"]['Sensor']["Mean"][4])) + ' & '
                 + str('%.2f'%(dict_objects['total_objects']['Baseline']["Mean"][12])) + ', ' + str('%.2f'%(dict_objects['total_objects']['Baseline']["Mean"][13])) + ' & '
                 + str('%.2f'%(dict_objects['total_objects']['Sensor']["Mean"][12])) + ', ' + str('%.2f'%(dict_objects['total_objects']['Sensor']["Mean"][13])) + ' & '
                 + str('%.2f'%(dict_objects['total_objects']['Baseline']["Mean"][14])) + ', ' + str('%.2f'%(dict_objects['total_objects']['Baseline']["Mean"][15])) + ' & '
                 + str('%.2f'%(dict_objects['total_objects']['Sensor']["Mean"][14])) + ', ' + str('%.2f'%(dict_objects['total_objects']['Sensor']["Mean"][15])) + ' & '

                 + str('%.2f'%(dict_objects["total_objects"]['Baseline']["Mean"][9])) + ', ' + str('%.2f'%(dict_objects["total_objects"]['Baseline']["Mean"][10])) + ' & '
                 + str('%.2f'%(dict_objects["total_objects"]['Sensor']["Mean"][9])) + ', ' + str('%.2f'%(dict_objects["total_objects"]['Sensor']["Mean"][10])))
        tf.write(r'\\')
        tf.write('\n')
        tf.write(r'\hline')
        tf.write('\n')
        tf.write(r'\end{tabular}')
        tf.write('\n')
        tf.write(r'\end{table*}')
        tf.write('\n')
        tf.write(r'\end{document}')


def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--results_directory', dest='results_directory', help='path to the results directory',
                        type=str, required=True)
    parser.add_argument('--ycb_directory', dest='ycb_directory', help='path to the ycb directory',
                        type=str, required=True)
    parser.add_argument('--json_directory', dest='json_directory', help='path to the json directory',
                        type=str, required=True)
    parser.add_argument('-l','--list', dest='gt',action='append')

    args = parser.parse_args()

    average_error = []
    average_rotation_error = []
    average_position_error = []
    ground_truth_position = np.array([float(args.gt[0]),
                                      float(args.gt[1]),
                                      float(args.gt[2])])

    with open(args.json_directory + 'contacts.json') as f:
        contacts_dict = json.load(f)

    with open(args.json_directory + "objects.json") as fo:
        list_objects = json.load(fo)

    average_error = np.array(average_error)
    average_rotation_error = np.array(average_rotation_error)
    average_position_error = np.array(average_position_error)

    error_dict = {"Best": [], "Best 5": [[], [], []], "Best 10": [[], [], []], "Mean": [], "Poses":[]}
    sensor_vs_baseline = {"Sensor": copy.deepcopy(error_dict), "Baseline": copy.deepcopy(error_dict)}

    dict_objects = {}
    for h in list_objects:
        dict_objects[h] = [copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline)]

    dict_objects["total_objects"] = copy.deepcopy(sensor_vs_baseline)

    method_list_path = ["results_sensor/", "results_baseline/"]
    method_list = ["Sensor", "Baseline"]
    triplet_list_path = ["triplet_1/","triplet_2/", "triplet_3/", "triplet_4/"]

    for key, _ in dict_objects.items():

        if key == 'total_objects':
            break
        for triplet in range(len(triplet_list_path)):
            for method in range(len(method_list)):
                arrays = np.load(args.results_directory + "/" + key + "/" + triplet_list_path[triplet] + method_list_path[method] + 'final_results.npz')
                all_rotations = arrays['total_poses']
                sum_errors = arrays['sum_errors']
                rotation_errors = arrays['pose_errors']
                position_errors = arrays['position_errors']
                ordered_poses = arrays['ordered_poses']

                for j in range(10):
                    index_best = int(ordered_poses[j])

                    if j == 0:
                        dict_objects[key][triplet][method_list[method]]["Best"].append(sum_errors[index_best])
                        dict_objects[key][triplet][method_list[method]]["Best"].append(position_errors[index_best] * 100)
                        dict_objects[key][triplet][method_list[method]]["Best"].append(rotation_errors[index_best])
                    if j < 5:

                        dict_objects[key][triplet][method_list[method]]["Best 5"][0].append(sum_errors[index_best])
                        dict_objects[key][triplet][method_list[method]]["Best 5"][1].append(position_errors[index_best] * 100)
                        dict_objects[key][triplet][method_list[method]]["Best 5"][2].append(rotation_errors[index_best])
                        angle = np.linalg.norm(all_rotations[index_best, 3:])
                        axis_angle = np.array([all_rotations[index_best, 0], all_rotations[index_best, 1], all_rotations[index_best, 2], all_rotations[index_best, 3]/angle, all_rotations[index_best, 4]/angle, all_rotations[index_best, 5]/angle, angle])
                        dict_objects[key][triplet][method_list[method]]["Poses"].append(axis_angle)
                        contacts_dict[key][triplet][method_list[method]][j]= int(contacts_dict[key][triplet][method_list[method]][j]/3) * 100
                    dict_objects[key][triplet][method_list[method]]["Best 10"][0].append(sum_errors[index_best])
                    dict_objects[key][triplet][method_list[method]]["Best 10"][1].append(position_errors[index_best] * 100)
                    dict_objects[key][triplet][method_list[method]]["Best 10"][2].append(rotation_errors[index_best])

                # Average of the 5 and 10 poses for the same triplet and contacts
                dict_objects[key][triplet][method_list[method]]["Mean"].append(dict_objects[key][triplet][method_list[method]]["Best"][0])
                dict_objects[key][triplet][method_list[method]]["Mean"].append(dict_objects[key][triplet][method_list[method]]["Best"][1])
                dict_objects[key][triplet][method_list[method]]["Mean"].append(dict_objects[key][triplet][method_list[method]]["Best"][2])
                dict_objects[key][triplet][method_list[method]]["Mean"].append(np.mean(np.array(dict_objects[key][triplet][method_list[method]]["Best 5"][0])))
                dict_objects[key][triplet][method_list[method]]["Mean"].append(np.mean(np.array(dict_objects[key][triplet][method_list[method]]["Best 5"][1])))
                dict_objects[key][triplet][method_list[method]]["Mean"].append(np.mean(np.array(dict_objects[key][triplet][method_list[method]]["Best 5"][2])))
                dict_objects[key][triplet][method_list[method]]["Mean"].append(np.mean(np.array(dict_objects[key][triplet][method_list[method]]["Best 10"][0])))
                dict_objects[key][triplet][method_list[method]]["Mean"].append(np.mean(np.array(dict_objects[key][triplet][method_list[method]]["Best 10"][1])))
                dict_objects[key][triplet][method_list[method]]["Mean"].append(np.mean(np.array(dict_objects[key][triplet][method_list[method]]["Best 10"][2])))
                dict_objects[key][triplet][method_list[method]]["Mean"].append(contacts_dict[key][triplet][method_list[method]][0])
                dict_objects[key][triplet][method_list[method]]["Mean"].append(np.mean(np.array(contacts_dict[key][triplet][method_list[method]][:5])))
                dict_objects[key][triplet][method_list[method]]["Mean"].append(np.mean(np.array(contacts_dict[key][triplet][method_list[method]])))


        # Average of the best, 5 and 10 poses for all the triplets
        for k in range(12):
            dict_objects[key][4]['Sensor']["Mean"].append(np.mean(np.array([dict_objects[key][0]['Sensor']["Mean"][k],
                                                                            dict_objects[key][1]['Sensor']["Mean"][k],
                                                                            dict_objects[key][2]['Sensor']["Mean"][k],
                                                                            dict_objects[key][3]['Sensor']["Mean"][k]])))
            dict_objects[key][4]['Baseline']["Mean"].append(np.mean(np.array([dict_objects[key][0]['Baseline']["Mean"][k],
                                                                              dict_objects[key][1]['Baseline']["Mean"][k],
                                                                              dict_objects[key][2]['Baseline']["Mean"][k],
                                                                              dict_objects[key][3]['Baseline']["Mean"][k]])))



        dict_objects[key][4]['Sensor']["Mean"].append(calculate_auc(key, np.concatenate((np.array([dict_objects[key][0]['Sensor']["Poses"][0]]),
                                                                                         np.array([dict_objects[key][1]['Sensor']["Poses"][0]]),
                                                                                         np.array([dict_objects[key][2]['Sensor']["Poses"][0]]),
                                                                                         np.array([dict_objects[key][3]['Sensor']["Poses"][0]])), axis=0), 4, args.ycb_directory, ground_truth_position, list_objects))

        dict_objects[key][4]['Baseline']["Mean"].append(calculate_auc(key, np.concatenate((np.array([dict_objects[key][0]['Baseline']["Poses"][0]]),
                                                                                           np.array([dict_objects[key][1]['Baseline']["Poses"][0]]),
                                                                                           np.array([dict_objects[key][2]['Baseline']["Poses"][0]]),
                                                                                           np.array([dict_objects[key][3]['Baseline']["Poses"][0]])), axis=0), 4, args.ycb_directory, ground_truth_position, list_objects))

        dict_objects[key][4]['Sensor']["Mean"].append(calculate_auc(key, np.concatenate((np.array(dict_objects[key][0]['Sensor']["Poses"]),
                                                                                         np.array(dict_objects[key][1]['Sensor']["Poses"]),
                                                                                         np.array(dict_objects[key][2]['Sensor']["Poses"]),
                                                                                         np.array(dict_objects[key][3]['Sensor']["Poses"])), axis=0), 20, args.ycb_directory, ground_truth_position, list_objects))

        dict_objects[key][4]['Baseline']["Mean"].append(calculate_auc(key, np.concatenate((np.array(dict_objects[key][0]['Baseline']["Poses"]),
                                                                                           np.array(dict_objects[key][1]['Baseline']["Poses"]),
                                                                                           np.array(dict_objects[key][2]['Baseline']["Poses"]),
                                                                                           np.array(dict_objects[key][3]['Baseline']["Poses"])), axis=0), 20, args.ycb_directory, ground_truth_position, list_objects))

        dict_objects[key][4]['Sensor']["Mean"].append(calculate_auc_rot(key, np.concatenate((np.array([dict_objects[key][0]['Sensor']["Poses"][0]]),
                                                                                             np.array([dict_objects[key][1]['Sensor']["Poses"][0]]),
                                                                                             np.array([dict_objects[key][2]['Sensor']["Poses"][0]]),
                                                                                             np.array([dict_objects[key][3]['Sensor']["Poses"][0]])), axis=0), 4, args.ycb_directory, ground_truth_position, list_objects))

        dict_objects[key][4]['Baseline']["Mean"].append(calculate_auc_rot(key, np.concatenate((np.array([dict_objects[key][0]['Baseline']["Poses"][0]]),
                                                                                               np.array([dict_objects[key][1]['Baseline']["Poses"][0]]),
                                                                                               np.array([dict_objects[key][2]['Baseline']["Poses"][0]]),
                                                                                               np.array([dict_objects[key][3]['Baseline']["Poses"][0]])), axis=0), 4, args.ycb_directory, ground_truth_position, list_objects))

        dict_objects[key][4]['Sensor']["Mean"].append(calculate_auc_rot(key, np.concatenate((np.array(dict_objects[key][0]['Sensor']["Poses"]),
                                                                                             np.array(dict_objects[key][1]['Sensor']["Poses"]),
                                                                                             np.array(dict_objects[key][2]['Sensor']["Poses"]),
                                                                                             np.array(dict_objects[key][3]['Sensor']["Poses"])), axis=0), 20, args.ycb_directory, ground_truth_position, list_objects))

        dict_objects[key][4]['Baseline']["Mean"].append(calculate_auc_rot(key, np.concatenate((np.array(dict_objects[key][0]['Baseline']["Poses"]),
                                                                                               np.array(dict_objects[key][1]['Baseline']["Poses"]),
                                                                                               np.array(dict_objects[key][2]['Baseline']["Poses"]),
                                                                                               np.array(dict_objects[key][3]['Baseline']["Poses"])), axis=0), 20, args.ycb_directory, ground_truth_position, list_objects))


    for k in range(16):
        arr_sensor = np.empty((0,1))
        arr_baseline = np.empty((0,1))
        for h in list_objects:
            arr_sensor = np.append(arr_sensor, np.array([[dict_objects[h][4]['Sensor']["Mean"][k]]]), 0)
            arr_baseline = np.append(arr_baseline, np.array([[dict_objects[h][4]['Baseline']["Mean"][k]]]), 0)

        dict_objects["total_objects"]["Sensor"]["Mean"].append(np.mean(arr_sensor))
        dict_objects["total_objects"]["Baseline"]["Mean"].append(np.mean(arr_baseline))
    create_table_all_objects(dict_objects, args.results_directory, list_objects)


if __name__ == '__main__':
    main()
