#!/bin/bash

python create_config_files.py --working_directory $1 --weights $2
mkdir $1/rotations
mkdir $1/remaining_poses
mkdir $1/depths
for object in 002_master_chef_can 004_sugar_box 006_mustard_bottle 007_tuna_fish_can 008_pudding_box 011_banana 019_pitcher_base 021_bleach_cleanser 036_wood_block 040_large_marker; do

    for triplet in triplet_1 triplet_2 triplet_3 triplet_4 ; do

        # Ours method
        python optimization.py --config_file_path $1/results_directory_pose_estimation/${object}/${triplet}/config_directory/config_total_test.ini --enable_selection true
        rm $1/rotations/*
        mv rotations_best* $1/rotations/

        rm $1/depths/*

        OMP_NUM_THREADS=10 $1/physX/build/script $1//results_directory_pose_estimation/${object}/${triplet}/config_directory/config_physx.ini

        python evaluate_poses.py --config_file_path $1/results_directory_pose_estimation/${object}/${triplet}/config_directory/config_total_test.ini

        cp landmarks.off final_results.npz $1/results_directory_pose_estimation/${object}/${triplet}/results_sensor/

        # Uncomment if you want to save the point clouds related to the best poses
        #python save_poses.py --config_file_path $1/results_directory_pose_estimation/${object}/${triplet}/config_directory/config_total_test.ini
        #cp -r remaining_poses $1/results_directory_pose_estimation/${object}/${triplet}/results_sensor/

        # Geometric baseline
        python optimization.py --config_file_path $1/results_directory_pose_estimation/${object}/${triplet}/config_directory/config_total_test.ini
        rm $1/rotations/*
        mv rotations_best* $1/rotations/

        rm $1/depths/*
        OMP_NUM_THREADS=10 $1/physX/build/script $1//results_directory_pose_estimation/${object}/${triplet}/config_directory/config_physx.ini

        python evaluate_poses.py --config_file_path $1/results_directory_pose_estimation/${object}/${triplet}/config_directory/config_total_test.ini

        cp landmarks.off final_results.npz $1/results_directory_pose_estimation/${object}/${triplet}/results_baseline/

        # Uncomment if you want to save the point clouds related to the best poses
        #python save_poses.py --config_file_path $1/results_directory_pose_estimation/${object}/${triplet}/config_directory/config_total_test.ini
        #cp -r remaining_poses $1/results_directory_pose_estimation/${object}/${triplet}/results_baseline/
    done
done

python final_results.py --results_directory $1/results_directory_pose_estimation/ --json_directory $1/json/ --ycb_directory $1/YCB_Video_Models/ -l 0.1 -l 0.0 -l 0.2
