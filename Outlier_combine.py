# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:25:42 2024

@author: gib445
"""

import os 
# Define the directory path
directory_path = r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\Desktop\PhD\Input_study'

# Define the file paths for the two text files
exclude_files_imu_path = os.path.join(directory_path, 'exclude_files_IMU.txt')
exclude_files_omc_path = os.path.join(directory_path, 'exclude_files_OMC.txt')
output_file_path = os.path.join(directory_path, 'outliers_IMU_OMC.txt')

# Read the numbers from the exclude_files_IMU
with open(exclude_files_imu_path, 'r') as imu_file:
    imu_numbers = set(map(int, imu_file.read().splitlines()))  # Read, split, and convert to integers

# Read the numbers from the exclude_files_OMC
with open(exclude_files_omc_path, 'r') as omc_file:
    omc_numbers = set(map(int, omc_file.read().splitlines()))  # Read, split, and convert to integers

# Combine the two sets and sort the unique numbers
combined_numbers = sorted(imu_numbers.union(omc_numbers))  # Union of both sets, sorted

# Write the combined and sorted numbers to the new file
with open(output_file_path, 'w') as output_file:
    for number in combined_numbers:
        output_file.write(f"{number}\n")

print(f"Combined and sorted numbers have been written to {output_file_path}")
