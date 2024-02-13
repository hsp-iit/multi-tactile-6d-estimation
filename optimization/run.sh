#!/bin/bash

python create_config_files_cleaned.py --json_folder ./json_folder --weights_path ./weights/model_real_back_norm.pth --working_directory .
mkdir rotations

for object in 002_master_chef_can 004_sugar_box 006_mustard_bottle 007_tuna_fish_can 008_pudding_box 011_banana 019_pitcher_base 021_bleach_cleanser 036_wood_block 040_large_marker; do

    for triplet in triplet_1 triplet_2 triplet_3 triplet_4 ; do

        cp results_directory_pose_estimation/${object}/${triplet}/config_directory/config_total_test.ini ./config_total.ini
        cp point_clouds/${object}/angles_database.txt ./angles_database.txt
        cp results_directory_pose_estimation/${object}/${triplet}/config_directory/config_physx.ini ./phys_example/config.txt

        # Ours method
        python optimization_git.py --config_file_path config_total.ini --enable_selection true
        rm rotations/*
        mv rotations_best* rotations/

        cd physX/build
        rm depths*

        OMP_NUM_THREADS=10 ./general_test ../config.txt
        cd ../../

        python evaluate_poses_test.py --config_file_path config_total.ini
        rm remaining_poses/*
        python save_poses.py --config_file_path config_total.ini
        cp landmarks.off final_results.npz results_directory_pose_estimation/${object}/${triplet}/results_sensor/
        cp -r remaining_poses results_directory_pose_estimation/${object}/${triplet}/results_sensor/


        # Geometric baseline
        python optimization_git.py --config_file_path config_total.ini
        rm rotations/*
        mv rotations_best* rotations/

        cd physX/build
        rm depths*
        OMP_NUM_THREADS=10 ./general_test ../config.txt
        cd ../../

        python evaluate_poses_test.py --config_file_path config_total.ini
        rm remaining_poses/*
        python save_poses.py --config_file_path config_total.ini
        cp -r remaining_poses results_directory_pose_estimation/${object}/${triplet}/results_baseline/
        cp final_results.npz results_directory_pose_estimation/${object}/${triplet}/results_baseline/

    done
done

python final_results.py --results_directory results_directory_pose_estimation