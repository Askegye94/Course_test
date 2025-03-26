# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 22:16:54 2024

@author: gib445
"""

import pandas as pd

# Path to your TRC file
file_path = r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\Desktop\Test_data\MarkerData\Trial5.trc'

# Read the TRC file and skip the header rows
with open(file_path, 'r') as file:
    lines = file.readlines()

# Extract column names
header_line = lines[3]  # The line containing column headers
columns = header_line.strip().split('\t')
print(columns)

# Read the numerical data into a DataFrame
data_lines = lines[6:]  # Data starts after the 6th line
data = [line.strip().split('\t') for line in data_lines]
df = pd.DataFrame(data)

# Handle empty columns and only retain marker names
filtered_columns = []
for i, col in enumerate(columns):
    if col.strip() != '':  # Skip empty columns
        filtered_columns.append(col.strip())

# Rename columns: for marker names, we create _X, _Y, _Z
final_columns = []
for col in filtered_columns:
    if col in ['Frame#', 'Time']:  # Keep special columns as is
        final_columns.append(col)
    else:
        # Add X, Y, Z suffixes for each marker
        final_columns.extend([f"{col}_X", f"{col}_Y", f"{col}_Z"])
        print(final_columns)

# Ensure the final columns match the data shape
if len(final_columns) != df.shape[1]:
    print(f"Warning: Adjusting columns from {len(final_columns)} to {df.shape[1]}")
    final_columns = final_columns[:df.shape[1]]

df.columns = final_columns

# Remove columns not related to the lower body and pelvis
columns_to_remove = [
    'Neck_X', 'Neck_Y', 'Neck_Z', 
    'RShoulder_X', 'RShoulder_Y', 'RShoulder_Z',
    'RElbow_X', 'RElbow_Y', 'RElbow_Z',
    'RWrist_X', 'RWrist_Y', 'RWrist_Z',
    'LShoulder_X', 'LShoulder_Y', 'LShoulder_Z',
    'LElbow_X', 'LElbow_Y', 'LElbow_Z',
    'LWrist_X', 'LWrist_Y', 'LWrist_Z',
    'r_shoulder_study_X', 'r_shoulder_study_Y', 'r_shoulder_study_Z',
    'L_shoulder_study_X', 'L_shoulder_study_Y', 'L_shoulder_study_Z',
    'C7_study_X', 'C7_study_Y', 'C7_study_Z',
    'r_lelbow_study_X', 'r_lelbow_study_Y', 'r_lelbow_study_Z',
    'r_melbow_study_X', 'r_melbow_study_Y', 'r_melbow_study_Z',
    'r_lwrist_study_X', 'r_lwrist_study_Y', 'r_lwrist_study_Z',
    'r_mwrist_study_X', 'r_mwrist_study_Y', 'r_mwrist_study_Z',
    'L_lelbow_study_X', 'L_lelbow_study_Y', 'L_lelbow_study_Z',
    'L_melbow_study_X', 'L_melbow_study_Y', 'L_melbow_study_Z',
    'L_lwrist_study_X', 'L_lwrist_study_Y', 'L_lwrist_study_Z',
    'L_mwrist_study_X', 'L_mwrist_study_Y', 'L_mwrist_study_Z'
]

# Drop the columns that are not related to the pelvis and lower body
df_lower_body = df.drop(columns=columns_to_remove)

# # Save the DataFrame to a CSV file
# output_path = "C:/Users/gib445/surfdrive - Larsen, A.G. (Aske Gye)@surfdrive.surf.nl/PHD Aske/Paper Ideas/OMC vs IMU vs Video/OpenCapData/MarkerData/test1_markerdata.csv"
# df_lower_body.to_csv(output_path, index=False)
# print(f"Marker data with axis names saved to {output_path}")

#%%
import pandas as pd

# Define file paths
mot_file = r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\Desktop\Test_data\OpenSimData\Kinematics\Trial5.mot'

# Function to read MOT file
def load_mot(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the line where column headers start (look for "endheader")
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_line = i + 1  # Column names start right after "endheader"
            break
    
    # Read column names
    columns = lines[header_line].strip().split()
    
    # Read numerical data
    data_lines = [line.strip().split() for line in lines[header_line + 1:]]
    
    # Convert to DataFrame
    df = pd.DataFrame(data_lines, columns=columns)
    
    # Convert numeric columns to float
    df = df.apply(pd.to_numeric, errors='ignore')
    
    return df

# Load the TRC and MOT data
df_mot = load_mot(mot_file)

print("\nMOT Data:")
print(df_mot.head())

# Save the DataFrame to a CSV file
output_path = r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\Desktop\Test_data\test1_markerdata.csv'
df_lower_body.to_csv(output_path, index=False)
output_path1 = r'C:\Users\gib445\OneDrive - Vrije Universiteit Amsterdam\Desktop\Test_data\test1_OpenSimdata.csv'
df_mot.to_csv(output_path1, index=False)
print(f"Marker data with axis names saved to {output_path}")

