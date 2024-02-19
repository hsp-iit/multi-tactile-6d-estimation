import argparse
import numpy as np
import pyquaternion as pyq
from jaxlie import SO3
import pandas as pd
import os
import copy
import json
from bop_pose_error import adi as bop_adi
from bop_pose_error import VOCap

class ADIEvaluator():

    def __init__(self, ycbv_models_path):

        object_names =\
        [
            '002_master_chef_can',
            '004_sugar_box',
            '006_mustard_bottle',
            '007_tuna_fish_can',
            '008_pudding_box',
            '011_banana',
            '019_pitcher_base',
            '021_bleach_cleanser',
            '036_wood_block',
            '040_large_marker'
        ]

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



def calculate_auc(object_string, poses, num_poses, ycb_video_models_path):
    evaluator = ADIEvaluator(ycb_video_models_path)
    gt = np.array([0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    gt = np.repeat(np.expand_dims(gt, 0), num_poses, axis=0)

    dists, auc = evaluator.adi_auc(object_string, gt, poses)

    return auc

def calculate_auc_rot(object_string, poses, num_poses, ycb_video_models_path, bool_val = False):
    evaluator = ADIEvaluator(ycb_video_models_path)
    poses[:,:3] = np.array([0.1, 0.0, 0.2])
    gt = np.array([0.1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    gt = np.repeat(np.expand_dims(gt, 0), num_poses, axis=0)

    dists, auc = evaluator.adi_auc(object_string, gt, poses)

    if bool_val == True:
        print(dists)

    return auc

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




def create_table_all_objects(dict_objects, results_directory):
    list_objects_tex = [r'002{\_}master{\_}chef{\_}can & ', r'004{\_}sugar{\_}box & ', r'006{\_}mustard{\_}bottle & ', r'007{\_}tuna{\_}fish{\_}can & ', r'008{\_}pudding{\_}box & ', r'011{\_}banana & ', r'019{\_}pitcher{\_}base & ', r'021{\_}bleach{\_}cleanser & ', r'036{\_}wood{\_}block & ' , r'040{\_}large{\_}marker & ']
    list_keys = ['002_master_chef_can',  '004_sugar_box', '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '011_banana','019_pitcher_base', '021_bleach_cleanser', '036_wood_block' ,'040_large_marker']
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
    parser.add_argument('--json_directory', dest='json_directory', help='path to the json directory',
                        type=str, required=True)
    parser.add_argument('--ycb_directory', dest='ycb_directory', help='path to the ycb directory',
                        type=str, required=True)

    args = parser.parse_args()

    average_error = []
    average_rotation_error = []
    average_position_error = []

    with open(args.json_directory + '/contacts.json') as f:
        contacts_dict = json.load(f)

    average_error = np.array(average_error)
    average_rotation_error = np.array(average_rotation_error)
    average_position_error = np.array(average_position_error)

    error_dict = {"Best": [], "Best 5": [[], [], []], "Best 10": [[], [], []], "Mean": [], "Poses":[]}
    sensor_vs_baseline = {"Sensor": copy.deepcopy(error_dict), "Baseline": copy.deepcopy(error_dict)}
    dict_objects = {"002_master_chef_can": [copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline)],
                    "004_sugar_box":[copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline)],
                    "006_mustard_bottle": [copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline)],
                    "007_tuna_fish_can": [copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline)],
                    "008_pudding_box": [copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline)],
                    "011_banana": [copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline)],
                    "019_pitcher_base": [copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline)],
                    "021_bleach_cleanser": [copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline)],
                    "036_wood_block": [copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline)],
                    "040_large_marker": [copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline), copy.deepcopy(sensor_vs_baseline)],
                    "total_objects": copy.deepcopy(sensor_vs_baseline)}

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
                    #contacts_dict[key][triplet][method_list[method]][j]= int(contacts_dict[key][triplet][method_list[method]][j]/3) * 100
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
                                                                                         np.array([dict_objects[key][3]['Sensor']["Poses"][0]])), axis=0), 4, args.ycb_directory))

        dict_objects[key][4]['Baseline']["Mean"].append(calculate_auc(key, np.concatenate((np.array([dict_objects[key][0]['Baseline']["Poses"][0]]),
                                                                                           np.array([dict_objects[key][1]['Baseline']["Poses"][0]]),
                                                                                           np.array([dict_objects[key][2]['Baseline']["Poses"][0]]),
                                                                                           np.array([dict_objects[key][3]['Baseline']["Poses"][0]])), axis=0), 4, args.ycb_directory))

        dict_objects[key][4]['Sensor']["Mean"].append(calculate_auc(key, np.concatenate((np.array(dict_objects[key][0]['Sensor']["Poses"]),
                                                                                         np.array(dict_objects[key][1]['Sensor']["Poses"]),
                                                                                         np.array(dict_objects[key][2]['Sensor']["Poses"]),
                                                                                         np.array(dict_objects[key][3]['Sensor']["Poses"])), axis=0), 20, args.ycb_directory))

        dict_objects[key][4]['Baseline']["Mean"].append(calculate_auc(key, np.concatenate((np.array(dict_objects[key][0]['Baseline']["Poses"]),
                                                                                           np.array(dict_objects[key][1]['Baseline']["Poses"]),
                                                                                           np.array(dict_objects[key][2]['Baseline']["Poses"]),
                                                                                           np.array(dict_objects[key][3]['Baseline']["Poses"])), axis=0), 20, args.ycb_directory))

        dict_objects[key][4]['Sensor']["Mean"].append(calculate_auc_rot(key, np.concatenate((np.array([dict_objects[key][0]['Sensor']["Poses"][0]]),
                                                                                             np.array([dict_objects[key][1]['Sensor']["Poses"][0]]),
                                                                                             np.array([dict_objects[key][2]['Sensor']["Poses"][0]]),
                                                                                             np.array([dict_objects[key][3]['Sensor']["Poses"][0]])), axis=0), 4, args.ycb_directory))

        dict_objects[key][4]['Baseline']["Mean"].append(calculate_auc_rot(key, np.concatenate((np.array([dict_objects[key][0]['Baseline']["Poses"][0]]),
                                                                                               np.array([dict_objects[key][1]['Baseline']["Poses"][0]]),
                                                                                               np.array([dict_objects[key][2]['Baseline']["Poses"][0]]),
                                                                                               np.array([dict_objects[key][3]['Baseline']["Poses"][0]])), axis=0), 4, args.ycb_directory))

        dict_objects[key][4]['Sensor']["Mean"].append(calculate_auc_rot(key, np.concatenate((np.array(dict_objects[key][0]['Sensor']["Poses"]),
                                                                                             np.array(dict_objects[key][1]['Sensor']["Poses"]),
                                                                                             np.array(dict_objects[key][2]['Sensor']["Poses"]),
                                                                                             np.array(dict_objects[key][3]['Sensor']["Poses"])), axis=0), 20, args.ycb_directory))

        dict_objects[key][4]['Baseline']["Mean"].append(calculate_auc_rot(key, np.concatenate((np.array(dict_objects[key][0]['Baseline']["Poses"]),
                                                                                               np.array(dict_objects[key][1]['Baseline']["Poses"]),
                                                                                               np.array(dict_objects[key][2]['Baseline']["Poses"]),
                                                                                               np.array(dict_objects[key][3]['Baseline']["Poses"])), axis=0), 20, args.ycb_directory))


    for k in range(16):
        dict_objects["total_objects"]["Sensor"]["Mean"].append(np.mean(np.array([dict_objects["002_master_chef_can"][4]['Sensor']["Mean"][k],
                                                                                 dict_objects["004_sugar_box"][4]['Sensor']["Mean"][k],
                                                                                 dict_objects["006_mustard_bottle"][4]['Sensor']["Mean"][k],
                                                                                 dict_objects["007_tuna_fish_can"][4]['Sensor']["Mean"][k],
                                                                                 dict_objects["008_pudding_box"][4]['Sensor']["Mean"][k],
                                                                                 dict_objects["011_banana"][4]['Sensor']["Mean"][k],
                                                                                 dict_objects["019_pitcher_base"][4]['Sensor']["Mean"][k],
                                                                                 dict_objects["021_bleach_cleanser"][4]['Sensor']["Mean"][k],
                                                                                 dict_objects["036_wood_block"][4]['Sensor']["Mean"][k],
                                                                                 dict_objects["040_large_marker"][4]['Sensor']["Mean"][k]])))
        dict_objects["total_objects"]["Baseline"]["Mean"].append(np.mean(np.array([dict_objects["002_master_chef_can"][4]['Baseline']["Mean"][k],
                                                                                   dict_objects["004_sugar_box"][4]['Baseline']["Mean"][k],
                                                                                   dict_objects["006_mustard_bottle"][4]['Baseline']["Mean"][k],
                                                                                   dict_objects["007_tuna_fish_can"][4]['Baseline']["Mean"][k],
                                                                                   dict_objects["008_pudding_box"][4]['Baseline']["Mean"][k],
                                                                                   dict_objects["011_banana"][4]['Baseline']["Mean"][k],
                                                                                   dict_objects["019_pitcher_base"][4]['Baseline']["Mean"][k],
                                                                                   dict_objects["021_bleach_cleanser"][4]['Baseline']["Mean"][k],
                                                                                   dict_objects["036_wood_block"][4]['Baseline']["Mean"][k],
                                                                                   dict_objects["040_large_marker"][4]['Baseline']["Mean"][k]])))


    create_table_all_objects(dict_objects, args.results_directory)


if __name__ == '__main__':
    main()
