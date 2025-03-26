# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 00:50:48 2024

@author: gib445
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras import layers
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib import gridspec
import math
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from scipy.stats import ttest_rel
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import ttest_rel

# Configuration
seed_value = 1
data_path = r'C:\Users\gib445\surfdrive - Larsen, A.G. (Aske Gye)@surfdrive.surf.nl\GitHub\VAE_stroke_longitudinal\3dPreparedData\\'
IMU_OMC_data_path = r'C:\Users\gib445\surfdrive - Larsen, A.G. (Aske Gye)@surfdrive.surf.nl\PHD Aske\Paper Ideas\OMC vs IMU vs Video\3dPreparedData'
model_save_path = r'C:\Users\gib445\surfdrive - Larsen, A.G. (Aske Gye)@surfdrive.surf.nl\GitHub\VAE_stroke_longitudinal\\'
window_length = 200
number_of_columns = 18
batch_size = 64
latent_features = 3  # Latent feature size
learning_rate = 3e-4  # Lower learning rate to improve stability
epochs = 50 # Number of training epochs

input1 = [20, 50]  # filters, kernel size
input2 = [10, 40]  # filters, kernel size
input3 = [2, 30]   # filters, kernel size
input4 = [80]      # units in Dense layer

columns_names = ['LKneeAngles_X', 'LKneeAngles_Y', 'LKneeAngles_Z',
                 'LHipAngles_X', 'LHipAngles_Y', 'LHipAngles_Z',
                 'LAnkleAngles_X', 'LAnkleAngles_Y', 'LAnkleAngles_Z',
                 'RKneeAngles_X', 'RKneeAngles_Y', 'RKneeAngles_Z',
                 'RHipAngles_X', 'RHipAngles_Y', 'RHipAngles_Z',
                 'RAnkleAngles_X', 'RAnkleAngles_Y', 'RAnkleAngles_Z']

# New Configuration Flags
use_saved_model = False  # Set this to True to load a saved model, False to train a new one
use_saved_weights = False  # Set this to True to load saved weights, False for random initialization

# Set random seed for reproducibility
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load Data
def load_data():
    data_files = {
        'data': 'stored_3D_other_data_withoutS01722_latentfeatures1_3_frequency_50.npy',
        'y': 'stored_y_3D_adapted_withoutS01722_latentfeatures1_3_frequency_50.npy',
        'groups': 'stored_3D_groupsplit_withoutS01722_latentfeatures1_3_frequency_50.npy',
        'test_data': 'stored_3D_dataAugmented_latentfeatures1_3_frequency_100.npy',
        'conditions': 'stored_conditions_latentfeatures1_3_frequency_100.npy',
        'participant_id': 'stored_participant_ids_latentfeatures1_3_frequency_100.npy',
        'week_numbers': 'stored_week_numbers_latentfeatures1_3_frequency_100.npy',       
        'groups_IMU_OMC': 'stored_3D_groupsplit_OMC_latentfeatures1_3_frequency_50.npy',
        'data_IMU': 'stored_3D_other_data_IMU_latentfeatures1_3_frequency_50.npy',
        'data_OMC': 'stored_3D_other_data_OMC_latentfeatures1_3_frequency_50.npy',
        'ids_IMU_OMC': 'stored_y_3D_adapted_OMC_latentfeatures1_3_frequency_50.npy',
        'ids_IMU_only': 'stored_y_3D_adapted_IMU_latentfeatures1_3_frequency_50.npy'

    }
    other_data = np.load(os.path.join(data_path, data_files['data']))
    y_adapted = np.load(os.path.join(data_path, data_files['y']))
    groupsplit = np.load(os.path.join(data_path, data_files['groups']))
    test_data = np.load(os.path.join(data_path, data_files['test_data']))
    conditions = np.load(os.path.join(data_path, data_files['conditions']))
    participant_id = np.load(os.path.join(data_path, data_files['participant_id']))
    week_numbers = np.load(os.path.join(data_path, data_files['week_numbers']))  
    groups_IMU_OMC = np.load(os.path.join(IMU_OMC_data_path, data_files['groups_IMU_OMC']))
    data_IMU = np.load(os.path.join(IMU_OMC_data_path, data_files['data_IMU']))
    data_OMC = np.load(os.path.join(IMU_OMC_data_path, data_files['data_OMC']))
    ids_IMU_OMC = np.load(os.path.join(IMU_OMC_data_path, data_files['ids_IMU_OMC']))
    ids_IMU_only = np.load(os.path.join(IMU_OMC_data_path, data_files['ids_IMU_only']))
    
    exclude_files = np.loadtxt(r'C:\Users\gib445\surfdrive - Larsen, A.G. (Aske Gye)@surfdrive.surf.nl\GitHub\VAE_stroke_longitudinal\exclude_files.txt')
    outliers = np.loadtxt(r'C:\Users\gib445\surfdrive - Larsen, A.G. (Aske Gye)@surfdrive.surf.nl\GitHub\VAE_stroke_longitudinal\outliers.txt')
    outliers_IMU_OMC = np.loadtxt(r'C:\Users\gib445\surfdrive - Larsen, A.G. (Aske Gye)@surfdrive.surf.nl\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Outliers\outliers_IMU_OMC.txt')
    exclude_indices = np.int_(exclude_files)
    exclude_outliers = np.int_(outliers)
    exclude_outliers_IMU_OMC = np.int_(outliers_IMU_OMC)
    
    other_data = np.delete(other_data, exclude_indices, axis=0)
    y_adapted = np.delete(y_adapted, exclude_indices, axis=0)
    groupsplit = np.delete(groupsplit, exclude_indices, axis=0)
    test_data = np.delete(test_data, exclude_outliers, axis=0)
    conditions = np.delete(conditions, exclude_outliers, axis=0)
    participant_id = np.delete(participant_id, exclude_outliers, axis=0)
    week_numbers = np.delete(week_numbers, exclude_outliers, axis=0)
    groups_IMU_OMC = np.delete(groups_IMU_OMC, exclude_outliers_IMU_OMC, axis=0)
    data_IMU = np.delete(data_IMU, exclude_outliers_IMU_OMC, axis=0)
    data_OMC = np.delete(data_OMC, exclude_outliers_IMU_OMC, axis=0)
    ids_IMU_OMC = np.delete(ids_IMU_OMC, exclude_outliers_IMU_OMC, axis=0)
    ids_IMU_only = np.delete(ids_IMU_only, exclude_outliers_IMU_OMC, axis=0)

    # Cast data to float32 to ensure consistency
    other_data = other_data.astype(np.float32)
    y_adapted = y_adapted.astype(np.float32)
    test_data = test_data.astype(np.float32)
    conditions = conditions.astype(np.float32)
    participant_id = participant_id.astype(np.float32)
    week_numbers = week_numbers.astype(np.float32)   
    groups_IMU_OMC = groups_IMU_OMC.astype(np.float32)
    data_IMU = data_IMU.astype(np.float32)
    data_OMC = data_OMC.astype(np.float32)

    return other_data, y_adapted, groupsplit, test_data, conditions, participant_id, week_numbers, groups_IMU_OMC, data_IMU, data_OMC, ids_IMU_OMC, ids_IMU_only

# Split Data into Training and Test Sets
def split_data(other_data, y_adapted, groupsplit, test_size=0.3):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed_value)
    train_idx, test_idx = next(gss.split(other_data, y_adapted, groups=groupsplit))
    
    X_train, X_test = other_data[train_idx], other_data[test_idx]
    y_train, y_test = y_adapted[train_idx], y_adapted[test_idx]
    groups_train, groups_test = groupsplit[train_idx], groupsplit[test_idx]
    
    return X_train, X_test, y_train, y_test, groups_train, groups_test

# Main script to load and split data
other_data, y_adapted, groupsplit,  test_data, conditions, participant_id, week_numbers, groups_IMU_OMC, data_IMU, data_OMC, ids_IMU_OMC, ids_IMU_only = load_data()
X_train, X_test, y_train, y_test, groups_train, groups_test = split_data(other_data, y_adapted, groupsplit)

#%%

# Remove data from participant 3
mask = participant_id != 3
participant_id = participant_id[mask]
test_data = test_data[mask]
conditions = conditions[mask]
week_numbers = week_numbers[mask]

# Print the shapes of the datasets
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)
print("Training groups shape:", groups_train.shape)
print("Test groups shape:", groups_test.shape)

# List of columns that should NOT be flipped
flipped_axis_training_test = [0, 1, 4, 7, 9, 11, 14, 17]
flipped_axis_IMU_OMC = [1, 5, 10, 14]

#TEST LONGITUDINAL SO THEY ARE CORRECT
#TEST LONGITUDINAL SO THEY ARE CORRECT
flipped_axis_longitudinal = [0, 2, 4, 7, 8, 9, 10, 16]
#TEST LONGITUDINAL SO THEY ARE CORRECT
#TEST LONGITUDINAL SO THEY ARE CORRECT

# Loop through all columns and flip only those not in the excluded list
for i in range(18):
    if i in flipped_axis_training_test:
        X_train[:, :, i] *= -1
        X_test[:, :, i] *= -1
        
# Loop through all columns and flip only those not in the excluded list
for i in range(18):
    if i in flipped_axis_IMU_OMC:
        data_IMU[:, :, i] *= -1
        data_OMC[:, :, i] *= -1                

# Loop through all columns and flip only those not in the excluded list
for i in range(18):
    if i in flipped_axis_longitudinal:
        test_data[:, :, i] *= -1

# Define the RMSE and NRMSE functions
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def nrmse(predictions, targets):
    return np.sqrt((((predictions - targets) / targets) ** 2).mean())

# Define the sampling function for latent features
def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    batch_size = tf.shape(distribution_variance)[0]
    random = tf.keras.backend.random_normal(shape=(batch_size, tf.shape(distribution_variance)[1]))
    return distribution_mean + tf.exp(0.5 * distribution_variance) * random

# Custom layer to calculate KL divergence
class KLDivergenceLayer(layers.Layer):
    def call(self, inputs):
        distribution_mean, distribution_variance = inputs
        kl_loss = 1 + distribution_variance - tf.square(distribution_mean) - tf.exp(distribution_variance)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5
        return kl_loss

# Define the CVAE model class
class CVAE(tf.keras.Model):
    def __init__(self, latent_features):
        super(CVAE, self).__init__()
        self.latent_features = latent_features

        # Encoder
        self.encoder_input = tf.keras.layers.Input(shape=(window_length, number_of_columns))

        # Unpacking the filter and kernel size from input1, input2, and input3
        encoder = tf.keras.layers.Conv1D(filters=input1[0], kernel_size=input1[1], activation='relu')(self.encoder_input)
        encoder = tf.keras.layers.Conv1D(filters=input2[0], kernel_size=input2[1], activation='relu')(encoder)
        encoder = tf.keras.layers.Conv1D(filters=input3[0], kernel_size=input3[1], activation='relu')(encoder)
        
        encoder = tf.keras.layers.Flatten()(encoder)
        encoder = tf.keras.layers.Dense(input4[0])(encoder)  # Only using the first element of input4

        self.mean = tf.keras.layers.Dense(latent_features, name='mean')(encoder)
        self.log_variance = tf.keras.layers.Dense(latent_features, name='log_variance')(encoder)
        self.latent_encoding = tf.keras.layers.Lambda(sample_latent_features)([self.mean, self.log_variance])

        self.encoder_model = tf.keras.Model(self.encoder_input, [self.mean, self.log_variance, self.latent_encoding])

        # Decoder
        self.decoder_input = tf.keras.layers.Input(shape=(latent_features,))
        decoder = tf.keras.layers.Dense(input4[0])(self.decoder_input)
        decoder = tf.keras.layers.Reshape((1, input4[0]))(decoder)

        # Using the same input variables for Conv1DTranspose layers
        decoder = tf.keras.layers.Conv1DTranspose(filters=input3[0], kernel_size=input3[1], activation='relu')(decoder)
        decoder = tf.keras.layers.Conv1DTranspose(filters=input2[0], kernel_size=input2[1], activation='relu')(decoder)
        decoder = tf.keras.layers.Conv1DTranspose(filters=input1[0], kernel_size=input1[1], activation='relu')(decoder)

        decoder_output = tf.keras.layers.Conv1DTranspose(filters=number_of_columns, kernel_size=83)(decoder)
        decoder_output = tf.keras.layers.LeakyReLU(negative_slope=0.1)(decoder_output)
        
        self.decoder_model = tf.keras.Model(self.decoder_input, decoder_output)

        # KL Divergence layer
        self.kl_loss_layer = KLDivergenceLayer()
        
    def encode(self, x):
        mean, logvar, _ = self.encoder_model(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder_model(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

# Define the loss function components
def get_loss(kl_loss_layer):
    
    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = tf.keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch * window_length * number_of_columns
    
    def total_loss(y_true, y_pred, distribution_mean, distribution_variance):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = kl_loss_layer([distribution_mean, distribution_variance])
        return reconstruction_loss_batch + kl_loss_batch
        
    return total_loss

# Instantiate the CVAE model
if use_saved_model and os.path.exists(os.path.join(model_save_path, 'my_model.keras')):
    print("Loading saved model...")
    cvae = tf.keras.models.load_model(os.path.join(model_save_path, 'my_model.keras'), custom_objects={'KLDivergenceLayer': KLDivergenceLayer, 'sample_latent_features': sample_latent_features})
else:
    cvae = CVAE(latent_features)
    if use_saved_weights and os.path.exists(os.path.join(model_save_path, 'model_weights.h5')):
        print("Loading saved weights...")
        cvae.load_weights(os.path.join(model_save_path, 'model_weights.h5'))
    else:
        print("Using random initialization...")

# Prepare optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

# Prepare the loss function using the model's mean and log variance
loss_fn = get_loss(cvae.kl_loss_layer)

# Define the training step
@tf.function
def train_step(model, x, optimizer, loss_fn):
    """Executes one training step and returns the loss."""
    with tf.GradientTape() as tape:
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        loss = loss_fn(x, x_logit, mean, logvar)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
if not use_saved_model:
    for epoch in range(1, epochs+1):
        for batch in range(0, len(X_train), batch_size):
            train_x = X_train[batch:batch + batch_size]
            loss = train_step(cvae, train_x, optimizer, loss_fn)
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

    # Save the model and weights after training
    cvae.save(os.path.join(model_save_path, 'my_model.keras'))
    cvae.save_weights(os.path.join(model_save_path, 'model_weights.weights.h5'))


#%%
# # Variables for subject and activity
# selected_subject = "S19"  # Change this to the desired subject
# selected_activity = "walk35"  # Change this to the desired activity
# # 5, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19
# # Step 1: Filter rows based on selected subject and activity
# matching_indices = [
#     i for i, (subject, activity) in enumerate(ids_IMU_OMC)
#     if subject == selected_subject and activity == selected_activity
# ]

# if len(matching_indices) == 0:
#     print(f"No data found for subject {selected_subject} and activity {selected_activity}")
# else:
#     # Step 2: Extract matching data
#     filtered_data = data_IMU[matching_indices]  # Shape: (filtered_rows, 200, 18)
#     filtered_titles = [f"{ids_IMU_OMC[idx, 0]}, {ids_IMU_OMC[idx, 1]}" for idx in matching_indices]

#     # Step 3: Plotting loop
#     rows_to_plot = 30  # Number of rows to display per figure
#     time_points = filtered_data.shape[1]

#     for start_idx in range(0, len(filtered_data), rows_to_plot):
#         end_idx = min(start_idx + rows_to_plot, len(filtered_data))
#         fig, axes = plt.subplots(6, 3, figsize=(15, 10))
#         axes = axes.flatten()

#         for i, row_idx in enumerate(range(start_idx, end_idx)):
#             data_row = filtered_data[row_idx]  # Shape: (200, 18)
#             title = filtered_titles[row_idx]

#             for col in range(data_row.shape[1]):  # Iterate over 18 columns
#                 ax = axes[col]
#                 ax.plot(data_row[:, col], label=f"Row {row_idx}")
#                 ax.set_title(f"Feature {col + 1}")
#                 ax.set_xlabel("Time")
#                 ax.set_ylabel("Value")

#             # Annotate with title
#             fig.suptitle(f"Data from: {title}", fontsize=16)

#         plt.tight_layout(rect=[0, 0, 1, 0.96])
#         plt.show()
#%%
# Extract latent features
encoded_test_data = []
for i in range(len(test_data)):
    visualData = np.expand_dims(test_data[i], axis=0)
    mean, logvar = cvae.encode(visualData)
    latent_encoding = cvae.reparameterize(mean, logvar)
    encoded_test_data.append(latent_encoding[0].numpy())

# Convert the list to a numpy array
encoded_test_data = np.array(encoded_test_data)

# Convert the data to a DataFrame for easier manipulation
df = pd.DataFrame(encoded_test_data, columns=[f'latent_{i}' for i in range(encoded_test_data.shape[1])])
df['participant_id'] = participant_id
df['week_number'] = week_numbers

# Group by participant_id and week_number and calculate the mean for each latent variable
mean_latent_features = df.groupby(['participant_id', 'week_number']).mean().reset_index()

# Filter indices for healthy participants
healthy_indices = [i for i, label in enumerate(y_adapted) if label == 1]
healthy_indices_test = [i for i, label in enumerate(y_test) if label == 1]

# Filter the other_data to include only healthy participants
healthy_data = other_data[healthy_indices]
healthy_data_test = X_test[healthy_indices_test]

# Extract latent features for the filtered healthy data
encoded_healthy_data = []
for i in range(len(healthy_data)):
    visualData = np.expand_dims(healthy_data[i], axis=0)
    mean, logvar = cvae.encode(visualData)
    latent_encoding = cvae.reparameterize(mean, logvar)
    encoded_healthy_data.append(latent_encoding[0].numpy())

# Convert the list to a numpy array
encoded_healthy_data = np.array(encoded_healthy_data)

# Calculate the mean of the latent features across all healthy entries
mean_healthy_latent_features = np.mean(encoded_healthy_data, axis=0)


#%%

# Initialize dictionaries to store means
mean_data = {
    # 'healthy_data': None,
    'data_IMU': None,
    'data_OMC': None
}

# Selected column indices and their names
selected_indices = [2, 5, 8]
selected_column_names = [columns_names[i] for i in selected_indices]

# Datasets
datasets = {
    # 'healthy_data': healthy_data,
    'data_IMU': data_IMU,
    'data_OMC': data_OMC
}

# Calculate means for each dataset
for name, data in datasets.items():
    # Extract columns 0, 4, and 7 from the last dimension
    selected_columns = data[:, :, selected_indices]
    
    # Compute the mean across the first dimension
    mean_values = np.mean(selected_columns, axis=0)  # Result is 200 x 3
    
    # Store the result
    mean_data[name] = mean_values

# Create a DataFrame for plotting
plot_data = pd.DataFrame()
for name, mean_values in mean_data.items():
    for col_idx, col_name in enumerate(selected_column_names):
        plot_data[f'{name}_{col_name}'] = mean_values[:, col_idx]

# Plot
plt.figure(figsize=(12, 8))
for column in plot_data.columns:
    plt.plot(plot_data[column], label=column.split('_', 1)[1])  # Extract the descriptive column name

plt.title("Mean Values Across Selected Columns")
plt.xlabel("Time Steps")
plt.ylabel("Mean Value")
plt.legend(title="Dataset and Column")
plt.grid()
plt.show()


#%%

# Define colors for the 4 weeks
colors = ['r', 'g', 'b', 'y']  # Red, Green, Blue, Yellow

# Get the unique participant IDs
participants = mean_latent_features['participant_id'].unique()

# Combine all latent feature data to calculate global min and max
all_latent_features = np.concatenate([encoded_test_data, encoded_healthy_data])

# Calculate global x, y, z limits
x_min, y_min, z_min = all_latent_features.min(axis=0)
x_max, y_max, z_max = all_latent_features.max(axis=0)

print(f"Global x limits: {x_min} to {x_max}")
print(f"Global y limits: {y_min} to {y_max}")
print(f"Global z limits: {z_min} to {z_max}")

# Define colors for the 4 weeks
colors = ['r', 'g', 'b', 'y']  # Red, Green, Blue, Yellow

# Get the unique participant IDs
participants = mean_latent_features['participant_id'].unique()

# Create a 3D plot for each participant
for participant in participants:
    # Filter data for the current participant
    participant_data = mean_latent_features[mean_latent_features['participant_id'] == participant]
    
    # Create a new figure for each participant
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the data for each week
    for week in range(4):
        week_data = participant_data[participant_data['week_number'] == week]
        if not week_data.empty:
            # Extract the latent features
            latent_features = week_data[['latent_0', 'latent_1', 'latent_2']].values
            ax.scatter(latent_features[:, 0], latent_features[:, 1], latent_features[:, 2], 
                       color=colors[week], label=f'Week {week}')
    
    # Plot the mean latent features of healthy participants
    ax.scatter(mean_healthy_latent_features[0], mean_healthy_latent_features[1], mean_healthy_latent_features[2],
               color='k', marker='x', s=100, label='Mean Healthy')
    
    # Set global limits for x, y, z axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Set plot labels and title
    ax.set_xlabel('Latent Feature 1')
    ax.set_ylabel('Latent Feature 2')
    ax.set_zlabel('Latent Feature 3')
    ax.set_title(f'Participant {participant}')
    ax.legend()
    
    plt.savefig(f'participant_{participant}_latent_features.svg', format='svg')

    # Show plot
    plt.show()
#%%

def plot_reconstruction(data, model, case=170, title_prefix='Original'):
    # Define joint names for labeling subplots
    joint_names = ['Left Knee', 'Left Hip', 'Left Ankle', 'Right Knee', 'Right Hip', 'Right Ankle']

    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(6, 2, figsize=(15, 12))

    # Define the x-axis ticks to represent percentage from 0 to 100
    x_ticks = np.linspace(0, 200, 6)  # 6 ticks (0%, 20%, 40%, 60%, 80%, 100%)
    x_tick_labels = ['0', '20', '40', '60', '80', '100']  # Corresponding percentage labels

    # Plot original data
    for i in range(6):
        axes[i, 0].plot(data[case, 0:200, i*3:(i+1)*3])
        axes[i, 0].set_title(f'{joint_names[i]} (Original)')
        axes[i, 0].legend(['X', 'Y', 'Z'], loc='upper right')  # Legend for X, Y, Z coordinates
        axes[i, 0].set_xticks(x_ticks)  # Set x-ticks
        axes[i, 0].set_xticklabels(x_tick_labels)  # Set x-tick labels as percentages

    # Reconstruct and plot the data
    visualDecodedData = np.expand_dims(data[case], axis=0)
    mean, logvar = model.encode(visualDecodedData)
    latent_encoding = model.reparameterize(mean, logvar)
    reconstructed_data = model.decode(latent_encoding).numpy()

    # Plot reconstructed data
    for i in range(6):
        axes[i, 1].plot(reconstructed_data[0, 0:200, i*3:(i+1)*3])
        axes[i, 1].set_title(f'{joint_names[i]} (Reconstructed)')
        axes[i, 1].legend(['X', 'Y', 'Z'], loc='upper right')  # Legend for X, Y, Z coordinates
        axes[i, 1].set_xticks(x_ticks)  # Set x-ticks
        axes[i, 1].set_xticklabels(x_tick_labels)  # Set x-tick labels as percentages

    # Adjust layout to make space for titles
    plt.tight_layout()
    plt.show()

# Example usage:
plot_reconstruction(X_test, cvae, case=170, title_prefix='Original')

#%%

# Global font size variables
TITLE_FONTSIZE = 26
LEGEND_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 16
POINT_SIZE = 70
HORIZONTAL_OFFSET = -1  # Horizontal offset for the plot
LABEL_PAD = 10  # Padding for axis labels

def generate_color_map(custom_labels, color_order):
    labels = list(custom_labels.keys())
    color_map = {label: color_order[i] for i, label in enumerate(labels)}
    return color_map

def plot_latent_space_3D(cvae, X_test, y_test, latent_features, custom_labels=None, label_name="Fall Risk", 
                         plot_title=None):
    if latent_features != 3:
        print("Latent features are not 3. Skipping 3D plot.")
        return

    # Set the default color order
    default_color_order = ['blue', 'red', 'green', 'yellow']
    
    # Generate the color map for the labels in custom_labels
    if custom_labels is None:
        custom_labels = {label: f'{label_name} {label}' for label in np.unique(y_test)}
    color_map = generate_color_map(custom_labels, default_color_order)

    # Set all unspecified labels to grey
    color_map_default = {label: 'grey' for label in np.unique(y_test)}
    color_map_default.update(color_map)  # Overwrite with specified colors

    encoded_points = []
    labels = []

    for i in range(len(X_test)):
        # Encode each test sample to get its latent space representation
        visualData = np.expand_dims(X_test[i], axis=0)
        mean, logvar = cvae.encode(visualData)
        latent_encoding = cvae.reparameterize(mean, logvar)
        encoded_points.append(latent_encoding[0].numpy())
        labels.append(y_test[i])

    # Convert to numpy array
    encoded_points = np.array(encoded_points)
    labels = np.array(labels)

    # Prepare the dataframe for easy handling
    df = pd.DataFrame(encoded_points, columns=['Latent Feature 1', 'Latent Feature 2', 'Latent Feature 3'])
    df['Label'] = labels

    # Apply color mapping
    df['Color'] = df['Label'].map(lambda x: color_map_default[x])

    # Plot in 3D
    fig = plt.figure(figsize=(12, 16))  # Increase plot size
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['Latent Feature 1'], df['Latent Feature 2'], df['Latent Feature 3'], 
                         c=df['Color'], edgecolors='white', s=POINT_SIZE)

    # Labeling the axes with padding
    ax.set_xlabel('Latent Feature 1', fontsize=AXIS_LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel('Latent Feature 2', fontsize=AXIS_LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_zlabel('Latent Feature 3', fontsize=AXIS_LABEL_FONTSIZE, labelpad=LABEL_PAD)

    # Auto-scale the axes to fit the data
    ax.auto_scale_xyz(
        [df['Latent Feature 1'].min(), df['Latent Feature 1'].max()],
        [df['Latent Feature 2'].min(), df['Latent Feature 2'].max()],
        [df['Latent Feature 3'].min(), df['Latent Feature 3'].max()]
    )

    # # Set custom plot title if provided
    # if plot_title is not None:
    #     plt.title(plot_title, fontsize=TITLE_FONTSIZE)
    # else:
    #     plt.title(f'3D Latent Space Representation ({label_name})', fontsize=TITLE_FONTSIZE)

    # Create a custom legend
    legend_handles = []
    for label, custom_label in custom_labels.items():
        color = color_map_default[label]
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=custom_label,
                                         markerfacecolor=color, markersize=10))

    # Include "Others" category if necessary
    if len(np.unique(labels)) > len(custom_labels):
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Others',
                                         markerfacecolor='grey', markersize=10))

    if legend_handles:  # Only show legend if there are labels to show
        ax.legend(handles=legend_handles, loc="upper right", fontsize=LEGEND_FONTSIZE)

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

    # Adjust subplot to accommodate horizontal offset
    plt.subplots_adjust(left=HORIZONTAL_OFFSET, right=0.9, 
                        top=0.9, bottom=0.1)

    plt.show()


#%%

# # Example of how to use this
custom_labels_classes = {0: 'Stroke', 1: 'Healthy'}
# custom_labels_participants = {2: 'ID 2', 34: 'ID 34', 58: 'ID 58', 72: 'ID72'}
# custom_labels_weeks = {0:'01', 1:'09', 2:'17', 3:'26'}
# custom_labels_conditions = {-1:'???',0: 'zonder', 1:'met'}
# custom_labels_participant_id = {1: 'ID 1', 5: 'ID 5', 9: 'ID 9', 11: 'ID 11'}

plot_latent_space_3D(cvae, X_test, y_test, latent_features=3, custom_labels=custom_labels_classes, label_name="Fall Risk", 
                      plot_title="Latent Space - Stroke/Healthy")
# plot_latent_space_3D(cvae, X_test, groups_test, latent_features=3, custom_labels=custom_labels_participants, label_name="Participants", 
#                       plot_title="Latent Space - Participants IDs")
# plot_latent_space_3D(cvae, test_data, week_numbers, latent_features=3, custom_labels=custom_labels_weeks, label_name="Weeks", 
#                       plot_title="Latent Space - Rehabilitation Weeks")
# plot_latent_space_3D(cvae, test_data, conditions, latent_features=3, custom_labels=custom_labels_conditions, label_name="Conditions", 
#                       plot_title="Latent Space - Experimental Conditions")
# plot_latent_space_3D(cvae, test_data, participant_id, latent_features=3, custom_labels=custom_labels_participant_id, label_name="Participant ID", 
#                       plot_title="Latent Space - Participant IDs")

#%%
def plot_latent_space_3D(cvae, data_IMU, data_OMC, X_test, latent_features, plot_title=None):
    if latent_features != 3:
        print("Latent features are not 3. Skipping 3D plot.")
        return

    # Set the default color order for IMU, OMC, and Other
    color_map_default = {0: 'blue', 1: 'red', 2: 'green'}  # 0 -> IMU, 1 -> OMC, 2 -> Other

    encoded_points = []
    labels = []  # We'll use this to differentiate between datasets

    # Encode data from IMU dataset
    for i in range(len(data_IMU)):
        visualData = np.expand_dims(data_IMU[i], axis=0)
        mean, logvar = cvae.encode(visualData)
        latent_encoding = cvae.reparameterize(mean, logvar)
        encoded_points.append(latent_encoding[0].numpy())
        labels.append(0)  # Label IMU data with 0

    # Encode data from OMC dataset
    for i in range(len(data_OMC)):
        visualData = np.expand_dims(data_OMC[i], axis=0)
        mean, logvar = cvae.encode(visualData)
        latent_encoding = cvae.reparameterize(mean, logvar)
        encoded_points.append(latent_encoding[0].numpy())
        labels.append(1)  # Label OMC data with 1

    # Encode data from Other dataset
    for i in range(len(X_test)):
        visualData = np.expand_dims(X_test[i], axis=0)
        mean, logvar = cvae.encode(visualData)
        latent_encoding = cvae.reparameterize(mean, logvar)
        encoded_points.append(latent_encoding[0].numpy())
        labels.append(2)  # Label Other data with 2

    # Convert to numpy array
    encoded_points = np.array(encoded_points)
    labels = np.array(labels)

    # Prepare the dataframe for easy handling
    df = pd.DataFrame(encoded_points, columns=['Latent Feature 1', 'Latent Feature 2', 'Latent Feature 3'])
    df['Label'] = labels

    # Apply color mapping
    df['Color'] = df['Label'].map(lambda x: color_map_default[x])

    # Plot in 3D
    fig = plt.figure(figsize=(12, 16))  # Increase plot size
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['Latent Feature 1'], df['Latent Feature 2'], df['Latent Feature 3'], 
                         c=df['Color'], edgecolors='white', s=POINT_SIZE)

    # Labeling the axes with padding
    ax.set_xlabel('Latent Feature 1', fontsize=AXIS_LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel('Latent Feature 2', fontsize=AXIS_LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_zlabel('Latent Feature 3', fontsize=AXIS_LABEL_FONTSIZE, labelpad=LABEL_PAD)

    # Auto-scale the axes to fit the data
    ax.auto_scale_xyz(
        [df['Latent Feature 1'].min(), df['Latent Feature 1'].max()],
        [df['Latent Feature 2'].min(), df['Latent Feature 2'].max()],
        [df['Latent Feature 3'].min(), df['Latent Feature 3'].max()]
    )

    # Create a custom legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label="IMU", markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label="OMC", markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label="Other", markerfacecolor='green', markersize=10)
    ]

    # Add the legend
    ax.legend(handles=legend_handles, loc="upper right", fontsize=LEGEND_FONTSIZE)

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

    # Adjust subplot to accommodate horizontal offset
    plt.subplots_adjust(left=HORIZONTAL_OFFSET, right=0.9, 
                        top=0.9, bottom=0.1)

    # Set plot title if needed
    if plot_title is not None:
        plt.title(plot_title, fontsize=TITLE_FONTSIZE)
    else:
        plt.title("Latent Space Representation (IMU vs OMC vs Other)", fontsize=TITLE_FONTSIZE)

    plt.show()


plot_latent_space_3D(cvae, data_IMU, data_OMC, X_test, latent_features=3, 
                     plot_title="Latent Space - IMU vs OMC vs Other")

#%%
# Get unique subject IDs and speeds
unique_subjects = np.unique(ids_IMU_OMC[:, 0])
unique_speeds = np.unique(ids_IMU_OMC[:, 1])

# Create an empty list to store the results for raw latent features
raw_latent_2d_array = []

# Create an empty list to store the results for mean latent features
mean_latent_2d_array = []

# Create an empty list to store the results for standard deviation latent features
std_latent_2d_array = []

# Loop through each subject and speed
for subject in unique_subjects:
    for speed in unique_speeds:
        # Find the indices for this subject and speed
        indices = np.where((ids_IMU_OMC[:, 0] == subject) & (ids_IMU_OMC[:, 1] == speed))[0]
        
        if len(indices) > 0:
            # Collect latent representations for IMU and OMC
            latent_imu = []
            latent_omc = []
            
            for idx in indices:
                # Encode data_IMU to latent space
                imu_visual_data = np.expand_dims(data_IMU[idx], axis=0)
                imu_mean, imu_logvar = cvae.encode(imu_visual_data)
                imu_latent = cvae.reparameterize(imu_mean, imu_logvar)
                imu_latent = imu_latent[0].numpy()  # Convert latent tensor to numpy array
                
                # Append raw latent IMU data
                raw_latent_2d_array.append([
                    subject,           # Participant ID
                    speed,             # Walk speed
                    'IMU',             # Source
                    imu_latent[0],     # Latent feature 1
                    imu_latent[1],     # Latent feature 2
                    imu_latent[2],     # Latent feature 3
                ])
                
                latent_imu.append(imu_latent)
                
                # Encode data_OMC to latent space
                omc_visual_data = np.expand_dims(data_OMC[idx], axis=0)
                omc_mean, omc_logvar = cvae.encode(omc_visual_data)
                omc_latent = cvae.reparameterize(omc_mean, omc_logvar)
                omc_latent = omc_latent[0].numpy()  # Convert latent tensor to numpy array
                
                # Append raw latent OMC data
                raw_latent_2d_array.append([
                    subject,           # Participant ID
                    speed,             # Walk speed
                    'OMC',             # Source
                    omc_latent[0],     # Latent feature 1
                    omc_latent[1],     # Latent feature 2
                    omc_latent[2],     # Latent feature 3
                ])
                
                latent_omc.append(omc_latent)
            
            # Calculate means of the latent variables
            imu_latent_mean = np.mean(latent_imu, axis=0)
            omc_latent_mean = np.mean(latent_omc, axis=0)
            
            # Append the IMU mean data row
            mean_latent_2d_array.append([
                subject,           # Participant ID
                speed,             # Walk speed
                'IMU',             # Source (IMU or OMC)
                imu_latent_mean[0],  # Latent feature 1
                imu_latent_mean[1],  # Latent feature 2
                imu_latent_mean[2],  # Latent feature 3
            ])
            
            # Append the OMC mean data row
            mean_latent_2d_array.append([
                subject,           # Participant ID
                speed,             # Walk speed
                'OMC',             # Source (IMU or OMC)
                omc_latent_mean[0],  # Latent feature 1
                omc_latent_mean[1],  # Latent feature 2
                omc_latent_mean[2],  # Latent feature 3
            ])
            
            # Calculate standard deviations of the latent variables
            imu_latent_std = np.std(latent_imu, axis=0)
            omc_latent_std = np.std(latent_omc, axis=0)
            
            # Append the IMU standard deviation data row
            std_latent_2d_array.append([
                subject,           # Participant ID
                speed,             # Walk speed
                'IMU',             # Source (IMU or OMC)
                imu_latent_std[0],  # Latent feature 1 (std deviation)
                imu_latent_std[1],  # Latent feature 2 (std deviation)
                imu_latent_std[2],  # Latent feature 3 (std deviation)
            ])
            
            # Append the OMC standard deviation data row
            std_latent_2d_array.append([
                subject,           # Participant ID
                speed,             # Walk speed
                'OMC',             # Source (IMU or OMC)
                omc_latent_std[0],  # Latent feature 1 (std deviation)
                omc_latent_std[1],  # Latent feature 2 (std deviation)
                omc_latent_std[2],  # Latent feature 3 (std deviation)
            ])


columns = ['participant_id', 'speed', 'source', 'latent_0', 'latent_1', 'latent_2']
# Create DataFrames from the NumPy arrays
raw_latent_df = pd.DataFrame(raw_latent_2d_array, columns=columns)
mean_latent_df = pd.DataFrame(mean_latent_2d_array, columns=columns)
std_latent_df = pd.DataFrame(std_latent_2d_array, columns=columns)

# Save the DataFrames to Excel files in model_save_path
raw_latent_file = f"{model_save_path}/raw_latent_2d_array.xlsx"
mean_latent_file = f"{model_save_path}/mean_latent_2d_array.xlsx"
std_latent_file = f"{model_save_path}/std_latent_2d_array.xlsx"

raw_latent_df.to_excel(raw_latent_file, index=False)
mean_latent_df.to_excel(mean_latent_file, index=False)
std_latent_df.to_excel(std_latent_file, index=False)

print(f"Files saved to:\n{raw_latent_file}\n{mean_latent_file}\n{std_latent_file}")


#%%

# Split the data into two DataFrames: one for OMC and one for IMU
omc_data = raw_latent_df[raw_latent_df['source'] == 'OMC'].reset_index(drop=True)
imu_data = raw_latent_df[raw_latent_df['source'] == 'IMU'].reset_index(drop=True)

# Create an empty list to store the results of the paired t-tests
ttest_results = []
correlation_results = []

# Perform the paired t-test for each latent variable (latent_0, latent_1, latent_2)
for i in range(3):  # latent_0, latent_1, latent_2
    t_stat, p_value = stats.ttest_rel(omc_data.iloc[:, i + 3], imu_data.iloc[:, i + 3])  # Skip the first 3 columns: participant_id, speed, source
    ttest_results.append({
        'latent_variable': f'latent_{i}',
        'p_value': p_value
    })

# Calculate the Pearson correlation for each latent variable (latent_0, latent_1, latent_2)
for i in range(3):  # latent_0, latent_1, latent_2
    # Calculate Pearson correlation coefficient and p-value
    correlation, p_value = pearsonr(omc_data.iloc[:, i + 3], imu_data.iloc[:, i + 3])  # Skip first 3 columns: participant_id, speed, source
    correlation_results.append({
        'latent_variable': f'latent_{i}',
        'correlation': correlation,
    })
    
# Convert the results to a DataFrame
ttest_df = pd.DataFrame(ttest_results)
correlation_df = pd.DataFrame(correlation_results)

# Set pandas to display numbers in normal (non-scientific) notation
pd.set_option('display.float_format', '{:.12f}'.format)

# Show the results
print(correlation_df)
print(ttest_df)

df = raw_latent_df.copy()

# Map 'source' to 'System' (IMU vs OMC)
df['System'] = df['source'].map({'IMU': 'IMU', 'OMC': 'OMC'})

# List of latent variables to analyze
latent_variables = ['latent_0', 'latent_1', 'latent_2']

# Create a list to store model results
model_results = []

# Run the mixed-effects model for each latent variable
for latent in latent_variables:
    # Define the formula for the model
    formula = f'{latent} ~ System * speed'  # Fixed effects: System, speed, and interaction
    
    # Fit the model
    model = mixedlm(formula, df, groups=df['participant_id'], re_formula="~1")
    
    # Fit the model and store results
    result = model.fit()
    
    # Store the results in the model_results list
    model_results.append({
        'latent_variable': latent,
        'fixed_effects': result.fe_params,
        'random_effects': result.random_effects,
        'summary': result.summary()
    })

# Display the results for each model
for result in model_results:
    print(f"Model for {result['latent_variable']}:")
    print(result['summary'])
    print("\n")



#%%

# Create an empty list to store the results for healthy raw latent features
raw_healthy_latent_2d_array = []

# Process all healthy data
for idx in range(len(healthy_data)):
    # Encode healthy_data to latent space
    healthy_visual_data = np.expand_dims(healthy_data[idx], axis=0)
    healthy_mean, healthy_logvar = cvae.encode(healthy_visual_data)
    healthy_latent = cvae.reparameterize(healthy_mean, healthy_logvar)
    healthy_latent = healthy_latent[0].numpy()  # Convert latent tensor to numpy array
    
    # Append only the latent features for healthy data
    raw_healthy_latent_2d_array.append([
        healthy_latent[0],  # Latent feature 1
        healthy_latent[1],  # Latent feature 2
        healthy_latent[2],  # Latent feature 3
    ])

# Define columns for the healthy latent features
healthy_columns = ['latent_0', 'latent_1', 'latent_2']

# Create a DataFrame from the healthy latent features
raw_healthy_latent_df = pd.DataFrame(raw_healthy_latent_2d_array, columns=healthy_columns)

# Save the healthy latent features to an Excel file
raw_healthy_latent_file = f"{model_save_path}/raw_healthy_latent_2d_array.xlsx"
raw_healthy_latent_df.to_excel(raw_healthy_latent_file, index=False)

print(f"Healthy latent features saved to:\n{raw_healthy_latent_file}")

# Create an empty list to store the results for healthy raw latent features
raw_healthy_latent_2d_array_test = []

# Process all healthy data
for idx in range(len(healthy_data_test)):
    # Encode healthy_data to latent space
    healthy_visual_data_test = np.expand_dims(healthy_data_test[idx], axis=0)
    healthy_mean_test, healthy_logvar_test = cvae.encode(healthy_visual_data_test)
    healthy_latent_test = cvae.reparameterize(healthy_mean_test, healthy_logvar_test)
    healthy_latent_test = healthy_latent_test[0].numpy()  # Convert latent tensor to numpy array
    
    # Append only the latent features for healthy data
    raw_healthy_latent_2d_array_test.append([
        healthy_latent_test[0],  # Latent feature 1
        healthy_latent_test[1],  # Latent feature 2
        healthy_latent_test[2],  # Latent feature 3
    ])

# Create a DataFrame from the healthy latent features
raw_healthy_latent_test_df = pd.DataFrame(raw_healthy_latent_2d_array_test, columns=healthy_columns)

# Save the healthy latent features to an Excel file
raw_healthy_latent_file_test = f"{model_save_path}/raw_healthy_latent_2d_array_test.xlsx"
raw_healthy_latent_test_df.to_excel(raw_healthy_latent_file_test, index=False)

print(f"Healthy latent features saved to:\n{raw_healthy_latent_file_test}")

# Create an empty list to store the results for healthy raw latent features
raw_x_test_latent_2d_array = []

# Process all healthy data
for idx in range(len(X_test)):
    # Encode healthy_data to latent space
    X_test_visual_data = np.expand_dims(X_test[idx], axis=0)
    healthy_mean, healthy_logvar = cvae.encode(X_test_visual_data)
    X_test_latent = cvae.reparameterize(healthy_mean, healthy_logvar)
    X_test_latent = X_test_latent[0].numpy()  # Convert latent tensor to numpy array
    
    # Append only the latent features for healthy data
    raw_x_test_latent_2d_array.append([
        X_test_latent[0],  # Latent feature 1
        X_test_latent[1],  # Latent feature 2
        X_test_latent[2],  # Latent feature 3
    ])

# Create a DataFrame from the healthy latent features
raw_x_test_latent_2d_array = pd.DataFrame(raw_x_test_latent_2d_array, columns=healthy_columns)

# Create an empty list to store the results for healthy raw latent features
raw_training_data_latent_2d_array = []

# Process all healthy data
for idx in range(len(X_test)):
    # Encode healthy_data to latent space
    training_data_visual_data = np.expand_dims(other_data[idx], axis=0)
    healthy_mean, healthy_logvar = cvae.encode(training_data_visual_data)
    training_data = cvae.reparameterize(healthy_mean, healthy_logvar)
    training_data = training_data[0].numpy()  # Convert latent tensor to numpy array
    
    # Append only the latent features for healthy data
    raw_training_data_latent_2d_array.append([
        training_data[0],  # Latent feature 1
        training_data[1],  # Latent feature 2
        training_data[2],  # Latent feature 3
    ])

# Create a DataFrame from the healthy latent features
raw_training_data_latent_2d_array = pd.DataFrame(raw_training_data_latent_2d_array, columns=healthy_columns)

#%%

def reconstruct_joint_angles(cvae, data):
    """
    Reconstruct joint angles from latent features using the VAE decoder.
    
    Parameters:
    - cvae: Trained CVAE model.
    - data: DataFrame containing latent features (latent_0, latent_1, latent_2).
    
    Returns:
    - Reconstructed joint angles as a numpy array of shape (num_samples, 200, 18).
    """
    # Extract latent features from the DataFrame
    latent_features = data[['latent_0', 'latent_1', 'latent_2']].values  # Shape: (num_samples, 3)
    
    # Decode latent features to reconstruct joint angles
    reconstructed_data = cvae.decode(latent_features).numpy()  # Decode and convert to numpy
    
    return reconstructed_data

# Apply reconstruction to both imu_data and omc_data
imu_reconstructed = reconstruct_joint_angles(cvae, imu_data)
omc_reconstructed = reconstruct_joint_angles(cvae, omc_data)

# Display shapes of reconstructed data
print("IMU Reconstructed Shape:", imu_reconstructed.shape)
print("OMC Reconstructed Shape:", omc_reconstructed.shape)

# Function to calculate mean and std for each joint at every timestep
def calculate_mean_std(reconstructed_data):
    mean_data = np.mean(reconstructed_data, axis=0)  # Mean across samples (axis 0)
    std_data = np.std(reconstructed_data, axis=0)    # Std across samples (axis 0)
    return mean_data, std_data

# Function to plot mean ± std for each joint angle
def plot_mean_std(imu_reconstructed, omc_reconstructed, columns_names, title_prefix):
    mean_imu, std_imu = calculate_mean_std(imu_reconstructed)
    mean_omc, std_omc = calculate_mean_std(omc_reconstructed)

    # Create a 6x3 subplot (6 rows, 3 columns for IMU and OMC comparison)
    fig, axes = plt.subplots(6, 3, figsize=(18, 18))

    # Add the main title for the entire figure
    fig.suptitle(f'{title_prefix} - Mean ± Std of Joint Angles', fontsize=16)

    # Plot each of the 18 joint angles
    for i, col_name in enumerate(columns_names):
        row = i // 3  # Determine the row for the subplot
        col = i % 3   # Determine the column for the subplot
        
        joint_index = i  # Index corresponding to the column in the data (0 to 17)

        # Plot IMU and OMC data in the same subplot (mean ± std)
        axes[row, col].plot(np.arange(200), mean_imu[:, joint_index], label=f'IMU {col_name}', color='blue')
        axes[row, col].fill_between(np.arange(200), 
                                    mean_imu[:, joint_index] - std_imu[:, joint_index], 
                                    mean_imu[:, joint_index] + std_imu[:, joint_index], 
                                    color='blue', alpha=0.3)

        axes[row, col].plot(np.arange(200), mean_omc[:, joint_index], label=f'OMC {col_name}', color='red', linestyle='--')
        axes[row, col].fill_between(np.arange(200), 
                                    mean_omc[:, joint_index] - std_omc[:, joint_index], 
                                    mean_omc[:, joint_index] + std_omc[:, joint_index], 
                                    color='red', alpha=0.3)

        # Set titles, labels, and legends
        axes[row, col].set_title(f'{col_name}')
        axes[row, col].set_xlabel('Time Step')
        axes[row, col].set_ylabel('Angle (degrees)')
        axes[row, col].legend(loc='upper right')

    # Adjust the layout to avoid overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust the top margin to make space for the title
    plt.show()

# Call the function to plot the mean ± std for the reconstructed data
plot_mean_std(imu_reconstructed, omc_reconstructed, columns_names, title_prefix='Reconstructed data')

# Call the function to plot the mean ± std for the input data (you can use your actual input data here)
plot_mean_std(data_IMU, data_OMC, columns_names, title_prefix='Input data')


#%%

# Compute range and percentiles for latent features
ranges = {}
percentiles = {}

for col in ['latent_0', 'latent_1', 'latent_2']:
    ranges[col] = (raw_healthy_latent_df[col].min(), raw_healthy_latent_df[col].max())
    percentiles[col] = np.percentile(raw_healthy_latent_df[col], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Function to reconstruct joint angles
def reconstruct_joint_angles_test(cvae, latent_features):
    """
    Reconstruct joint angles from latent features using the VAE decoder.
    """
    reconstructed_data = cvae.decode(latent_features)  # Decode latent features
    return reconstructed_data

# Function to plot reconstructions with percentiles overlaid in the same subplot
def plot_reconstructions_combined_with_colors(latent_name, fixed_values, variable_percentiles, reconstructions, title_prefix):
    """
    Plots the reconstructed joint angles with percentiles overlaid in the same subplot, using a gradient of red colors.
    """
    # Create a colormap for shades of red
    cmap = cm.get_cmap('Reds', len(variable_percentiles))
    colors = [cmap(i) for i in range(len(variable_percentiles))]

    plt.figure(figsize=(15, 10))
    for joint_idx, joint_name in enumerate(columns_names):
        plt.subplot(6, 3, joint_idx + 1)
        for i, (p, data) in enumerate(zip(variable_percentiles, reconstructions)):
            plt.plot(data[0, :, joint_idx], color=colors[i], label=f'{p}th')  # Plot with color corresponding to percentile
        plt.title(joint_name)
        plt.xlabel('Time Steps')
        plt.ylabel('Angle')
    
    # Add a single legend for the entire figure
    # handles = [plt.Line2D([0], [0], color=colors[i], lw=2, label=f'{p}th Percentile') for i, p in enumerate(variable_percentiles)]
    # plt.figlegend(handles=handles, loc='lower center', ncol=len(variable_percentiles), fontsize='medium', title=f'{latent_name} Percentiles')
    plt.suptitle(f'{title_prefix} for {latent_name} with varying values', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])  # Adjust layout to make space for the legend
    plt.show()

# Loop through each latent feature
for variable_latent in ['latent_0', 'latent_1', 'latent_2']:
    # Fix the other two latents at their 50th percentile
    fixed_values = {
        'latent_0': percentiles['latent_0'][2],
        'latent_1': percentiles['latent_1'][2],
        'latent_2': percentiles['latent_2'][2]
    }
    fixed_values.pop(variable_latent)  # Remove the one being varied

    # Get the percentiles for the current variable latent
    variable_percentiles = percentiles[variable_latent]

    # Prepare reconstructions for the variable latent
    reconstructions = []
    for value in variable_percentiles:
        # Create a NumPy array with fixed and variable values
        latent_features = np.array([[fixed_values.get('latent_0', value),
                                      fixed_values.get('latent_1', value),
                                      fixed_values.get('latent_2', value)]])
        # Reconstruct joint angles
        reconstructed_data = reconstruct_joint_angles_test(cvae, latent_features)
        reconstructions.append(reconstructed_data)

    # Plot reconstructions for the current variable latent
    plot_reconstructions_combined_with_colors(
        latent_name=variable_latent,
        fixed_values=fixed_values,
        variable_percentiles=variable_percentiles,
        reconstructions=reconstructions,
        title_prefix='Joint Angles'
    )



#%%

def calculate_rmse(data_imu, data_omc, imu_reconstructed, omc_reconstructed, columns_names):
    rmse_results = {}

    for i, col_name in enumerate(columns_names):
        # Get the IMU and OMC data for the current joint angle (column)
        imu = data_imu[:, :, i]  # Shape: (5230, 200)
        omc = data_omc[:, :, i]  # Shape: (5230, 200)

        # Calculate error for each timepoint (along the 0th axis)
        error = imu - omc  # Error between IMU and OMC

        # Calculate MSE for each timepoint
        mse = np.mean(np.square(error), axis=0)  # Mean squared error for each timepoint

        # Calculate RMSE for each timepoint (square root of MSE)
        rmse = np.sqrt(mse)  # RMSE for each timepoint

        # Store RMSE for this column (joint angle)
        rmse_results[col_name] = {
            'imu_omc_rmse': rmse
        }

        # Now for the reconstructed data:
        imu_reconstructed_col = imu_reconstructed[:, :, i]  # Shape: (5230, 200)
        omc_reconstructed_col = omc_reconstructed[:, :, i]  # Shape: (5230, 200)

        # Calculate error for each timepoint for the reconstructed data
        error_reconstructed = imu_reconstructed_col - omc_reconstructed_col  # Error between reconstructed IMU and OMC

        # Calculate MSE for each timepoint for the reconstructed data
        mse_reconstructed = np.mean(np.square(error_reconstructed), axis=0)  # Mean squared error for each timepoint

        # Calculate RMSE for the reconstructed data (square root of MSE)
        rmse_reconstructed = np.sqrt(mse_reconstructed)  # RMSE for each timepoint

        # Store RMSE for the reconstructed column
        rmse_results[col_name].update({
            'reconstructed_imu_omc_rmse': rmse_reconstructed
        })

    # Print the RMSE results for each column (joint angle)
    for col_name, rmse_data in rmse_results.items():
        print(f"Column: {col_name}")
        print(f"RMSE (IMU vs OMC): {np.mean(rmse_data['imu_omc_rmse']):.4f}")
        print(f"RMSE (Reconstructed IMU vs OMC): {np.mean(rmse_data['reconstructed_imu_omc_rmse']):.4f}")
        print("-" * 50)

# Example usage:
calculate_rmse(data_IMU, data_OMC, imu_reconstructed, omc_reconstructed, columns_names)


#%%
def calculate_correlation_coefficient(data_imu, data_omc, imu_reconstructed, omc_reconstructed, columns_names):
    correlation_results = {}

    for i, col_name in enumerate(columns_names):
        # Get the IMU and OMC data for the current joint angle
        imu = data_imu[:, :, i]  # Shape: (5230, 200)
        omc = data_omc[:, :, i]  # Shape: (5230, 200)
        
        # Calculate the correlation coefficient between IMU and OMC data
        imu_omc_corr = np.corrcoef(imu.flatten(), omc.flatten())[0, 1]
        
        # Get the reconstructed IMU and OMC data for the current joint angle
        imu_reconstructed_col = imu_reconstructed[:, :, i]  # Shape: (5230, 200)
        omc_reconstructed_col = omc_reconstructed[:, :, i]  # Shape: (5230, 200)

        # Calculate the correlation coefficient between reconstructed IMU and OMC data
        imu_reconstructed_omc_corr = np.corrcoef(imu_reconstructed_col.flatten(), omc_reconstructed_col.flatten())[0, 1]

        # Store the results in the dictionary
        correlation_results[col_name] = {
            'imu_omc_corr': imu_omc_corr,
            'imu_reconstructed_omc_corr': imu_reconstructed_omc_corr
        }

    # Print the correlation results for each column
    for col_name, corrs in correlation_results.items():
        print(f"Column: {col_name}")
        print(f"IMU vs OMC Correlation: {corrs['imu_omc_corr']:.4f}")
        print(f"Reconstructed IMU vs OMC Correlation: {corrs['imu_reconstructed_omc_corr']:.4f}")
        print("-" * 50)

# Example usage:
calculate_correlation_coefficient(data_IMU, data_OMC, imu_reconstructed, omc_reconstructed, columns_names)

#%%

mean_latent_features = pd.DataFrame(mean_latent_2d_array, columns=columns)

# Convert speed to numerical order
mean_latent_features['speed_order'] = mean_latent_features['speed'].str.extract(r'(\d+)').astype(int)

# Determine number of participants
unique_participants = mean_latent_features['participant_id'].unique()
num_participants = len(unique_participants)
cols = 4  # Number of columns in the subplot grid
rows = math.ceil(num_participants / cols)  # Calculate rows needed

# Create the figure
fig = plt.figure(figsize=(20, rows * 5))  # Adjust figure size based on number of rows

# Loop through each participant and create a subplot
for idx, participant_id in enumerate(unique_participants, start=1):
    ax = fig.add_subplot(rows, cols, idx, projection='3d')
    
    # Filter data for the participant
    participant_data = mean_latent_features[mean_latent_features['participant_id'] == participant_id]
    
    # Separate IMU and OMC data
    imu_data = participant_data[participant_data['source'] == 'IMU'].sort_values(by='speed_order')
    omc_data = participant_data[participant_data['source'] == 'OMC'].sort_values(by='speed_order')
    
    # Plot IMU data points
    ax.scatter(imu_data['latent_0'], imu_data['latent_1'], imu_data['latent_2'], 
               c='blue', label='IMU', s=50)
    
    # Plot OMC data points
    ax.scatter(omc_data['latent_0'], omc_data['latent_1'], omc_data['latent_2'], 
               c='orange', label='OMC', s=50)
    
    # Connect IMU and OMC data points across speeds
    for speed in imu_data['speed_order']:
        imu_point = imu_data[imu_data['speed_order'] == speed][['latent_0', 'latent_1', 'latent_2']].values
        omc_point = omc_data[omc_data['speed_order'] == speed][['latent_0', 'latent_1', 'latent_2']].values
        if imu_point.size > 0 and omc_point.size > 0:
            ax.plot(
                [imu_point[0][0], omc_point[0][0]],
                [imu_point[0][1], omc_point[0][1]],
                [imu_point[0][2], omc_point[0][2]],
                linestyle='--', color='gray', alpha=0.7
            )
    
    # Add axis labels
    ax.set_xlabel("Latent 0")
    ax.set_ylabel("Latent 1")
    ax.set_zlabel("Latent 2")
    
    # Add title for each subplot
    ax.set_title(f"Participant {participant_id}")
    
    # Legend
    ax.legend()

# Adjust layout to avoid overlap
plt.tight_layout()

# Add a main title for the figure
plt.suptitle("3D Latent Feature Plot for Each Participant", fontsize=16, y=1.02)

# Show the plot
plt.show()
#%%

# Function to generate 3D ellipsoids
def get_3d_ellipse(mean, std, scale=1.0, n_points=100):
    """
    Generate points for a 3D ellipse (ellipsoid) based on mean and standard deviation.

    Args:
    - mean: 1D array-like of shape (3,) representing the mean coordinates.
    - std: 1D array-like of shape (3,) representing the standard deviations.
    - scale: Float, scaling factor for the ellipsoid size.
    - n_points: Number of points to sample along each axis.

    Returns:
    - x, y, z: Coordinates of the ellipsoid points.
    """
    # Generate a sphere
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Scale the sphere by standard deviations
    x = scale * std[0] * x
    y = scale * std[1] * y
    z = scale * std[2] * z
    
    # Translate by the mean
    x += mean[0]
    y += mean[1]
    z += mean[2]
    
    return x, y, z

# Determine number of participants
unique_participants = mean_latent_df['participant_id'].unique()
num_participants = len(unique_participants)
cols = 4  # Number of columns in the subplot grid
rows = math.ceil(num_participants / cols)  # Calculate rows needed

# Create the figure
fig = plt.figure(figsize=(20, rows * 5))  # Adjust figure size based on number of rows

# Transparency variable for healthy data blobs
healthy_alpha = 0.05
imu_omc_alpha = 0.4

# Loop through each participant and create a subplot
for idx, participant_id in enumerate(unique_participants, start=1):
    ax = fig.add_subplot(rows, cols, idx, projection='3d')
    
    # Filter data for the participant
    mean_data = mean_latent_df[mean_latent_df['participant_id'] == participant_id]
    std_data = std_latent_df[std_latent_df['participant_id'] == participant_id]
    
    # Separate IMU and OMC data
    imu_mean = mean_data[mean_data['source'] == 'IMU'].sort_values(by='speed')
    omc_mean = mean_data[mean_data['source'] == 'OMC'].sort_values(by='speed')
    imu_std = std_data[std_data['source'] == 'IMU'].sort_values(by='speed')
    omc_std = std_data[std_data['source'] == 'OMC'].sort_values(by='speed')
    
    # Plot IMU data points and shaded areas
    ax.scatter(imu_mean['latent_0'], imu_mean['latent_1'], imu_mean['latent_2'], 
               c='blue', label='IMU', s=50)
    for i in range(len(imu_mean)):
        x, y, z = get_3d_ellipse(
            imu_mean.iloc[i][['latent_0', 'latent_1', 'latent_2']].values,
            imu_std.iloc[i][['latent_0', 'latent_1', 'latent_2']].values
        )
        ax.plot_surface(x, y, z, color='blue', alpha=imu_omc_alpha)
    
    # Plot OMC data points and shaded areas
    ax.scatter(omc_mean['latent_0'], omc_mean['latent_1'], omc_mean['latent_2'], 
               c='orange', label='OMC', s=50)
    for i in range(len(omc_mean)):
        x, y, z = get_3d_ellipse(
            omc_mean.iloc[i][['latent_0', 'latent_1', 'latent_2']].values,
            omc_std.iloc[i][['latent_0', 'latent_1', 'latent_2']].values
        )
        ax.plot_surface(x, y, z, color='orange', alpha=imu_omc_alpha)
    
    # Connect IMU and OMC data points across speeds
    for speed in imu_mean['speed']:
        imu_point = imu_mean[imu_mean['speed'] == speed][['latent_0', 'latent_1', 'latent_2']].values
        omc_point = omc_mean[omc_mean['speed'] == speed][['latent_0', 'latent_1', 'latent_2']].values
        if imu_point.size > 0 and omc_point.size > 0:
            ax.plot(
                [imu_point[0][0], omc_point[0][0]],
                [imu_point[0][1], omc_point[0][1]],
                [imu_point[0][2], omc_point[0][2]],
                linestyle='--', color='gray', alpha=0.8
            )
    
    # Add healthy data blobs in light grey
    ax.scatter(raw_healthy_latent_test_df['latent_0'], raw_healthy_latent_test_df['latent_1'], raw_healthy_latent_test_df['latent_2'], 
               c='lightgrey', s=10, alpha=healthy_alpha, label='Healthy Data' if idx == 1 else "")
    
    # Add axis labels
    ax.set_xlabel("Latent 0")
    ax.set_ylabel("Latent 1")
    ax.set_zlabel("Latent 2")
    
    # Add title for each subplot
    ax.set_title(f"Participant {participant_id}")
    
    # Legend
    if idx == 1:  # Add legend only to the first subplot to avoid clutter
        ax.legend()

# Adjust layout to avoid overlap
plt.tight_layout()

# Add a main title for the figure
plt.suptitle("3D Latent Feature Plot with Standard Deviations and Healthy Data", fontsize=16, y=1.02)

# Show the plot
plt.show()

#%%
# Function to generate 3D ellipsoids
def get_3d_ellipse(mean, std, scale=1.0, n_points=100):
    """
    Generate points for a 3D ellipse (ellipsoid) based on mean and standard deviation.

    Args:
    - mean: 1D array-like of shape (3,) representing the mean coordinates.
    - std: 1D array-like of shape (3,) representing the standard deviations.
    - scale: Float, scaling factor for the ellipsoid size.
    - n_points: Number of points to sample along each axis.

    Returns:
    - x, y, z: Coordinates of the ellipsoid points.
    """
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    x = scale * std[0] * x
    y = scale * std[1] * y
    z = scale * std[2] * z
    
    x += mean[0]
    y += mean[1]
    z += mean[2]
    
    return x, y, z

# Select the two participants
selected_participants = ['S12', 'S16']

# Create the figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Transparency variable for healthy data blobs
healthy_alpha = 0.05
imu_omc_alpha = 0.4

for participant_id in selected_participants:
    # Filter data for the participant
    mean_data = mean_latent_df[mean_latent_df['participant_id'] == participant_id]
    std_data = std_latent_df[std_latent_df['participant_id'] == participant_id]
    
    # Separate IMU and OMC data
    imu_mean = mean_data[mean_data['source'] == 'IMU'].sort_values(by='speed')
    omc_mean = mean_data[mean_data['source'] == 'OMC'].sort_values(by='speed')
    imu_std = std_data[std_data['source'] == 'IMU'].sort_values(by='speed')
    omc_std = std_data[std_data['source'] == 'OMC'].sort_values(by='speed')
    
    # Plot IMU data points and shaded areas
    for i in range(len(imu_mean)):
        speed_label = f"IMU ({participant_id}), Speed {imu_mean.iloc[i]['speed']}"
        x, y, z = get_3d_ellipse(
            imu_mean.iloc[i][['latent_0', 'latent_1', 'latent_2']].values,
            imu_std.iloc[i][['latent_0', 'latent_1', 'latent_2']].values
        )
        ax.plot_surface(x, y, z, alpha=imu_omc_alpha, label=speed_label)
    
    # Plot OMC data points and shaded areas
    for i in range(len(omc_mean)):
        speed_label = f"OMC ({participant_id}), Speed {omc_mean.iloc[i]['speed']}"
        x, y, z = get_3d_ellipse(
            omc_mean.iloc[i][['latent_0', 'latent_1', 'latent_2']].values,
            omc_std.iloc[i][['latent_0', 'latent_1', 'latent_2']].values
        )
        ax.plot_surface(x, y, z, alpha=imu_omc_alpha, label=speed_label)

    # Connect IMU and OMC data points across speeds
    for speed in imu_mean['speed']:
        imu_point = imu_mean[imu_mean['speed'] == speed][['latent_0', 'latent_1', 'latent_2']].values
        omc_point = omc_mean[omc_mean['speed'] == speed][['latent_0', 'latent_1', 'latent_2']].values
        if imu_point.size > 0 and omc_point.size > 0:
            ax.plot(
                [imu_point[0][0], omc_point[0][0]],
                [imu_point[0][1], omc_point[0][1]],
                [imu_point[0][2], omc_point[0][2]],
                linestyle='--', color='gray', alpha=0.8
            )

# Add healthy data blobs in light grey
ax.scatter(raw_x_test_latent_2d_array['latent_0'], raw_x_test_latent_2d_array['latent_1'], raw_x_test_latent_2d_array['latent_2'], 
           c='lightgrey', s=10, alpha=healthy_alpha, label='Validation data')

# Add axis labels
ax.set_xlabel("Latent 0")
ax.set_ylabel("Latent 1")
ax.set_zlabel("Latent 2")

# Add title
ax.set_title(f"3D Latent Feature Plot for {selected_participants[0]} and {selected_participants[1]}", fontsize=16)

# Add legend
ax.legend()

save_path = r"C:\Users\gib445\surfdrive - Larsen, A.G. (Aske Gye)@surfdrive.surf.nl\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Figures\3D_latent_plot.svg"
fig.savefig(save_path, format='svg', bbox_inches='tight')

# Show the plot
plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np

# Function to generate 3D ellipsoids
def get_3d_ellipse(mean, std, scale=1.0, n_points=100):
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    x = scale * std[0] * x
    y = scale * std[1] * y
    z = scale * std[2] * z

    x += mean[0]
    y += mean[1]
    z += mean[2]

    return x, y, z

# Select the two participants
selected_participants = ['S12', 'S16']

# Transparency variables
healthy_alpha = 0.05
imu_omc_alpha = 0.4

# Initialize variables to store axis limits
x_min, x_max = np.inf, -np.inf
y_min, y_max = np.inf, -np.inf
z_min, z_max = np.inf, -np.inf

# Create the first figure (for participants)
fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_subplot(111, projection='3d')

for participant_id in selected_participants:
    # Filter data for the participant
    mean_data = mean_latent_df[mean_latent_df['participant_id'] == participant_id]
    std_data = std_latent_df[std_latent_df['participant_id'] == participant_id]

    # Separate IMU and OMC data
    imu_mean = mean_data[mean_data['source'] == 'IMU'].sort_values(by='speed')
    omc_mean = mean_data[mean_data['source'] == 'OMC'].sort_values(by='speed')
    imu_std = std_data[std_data['source'] == 'IMU'].sort_values(by='speed')
    omc_std = std_data[std_data['source'] == 'OMC'].sort_values(by='speed')

    # Plot IMU data points and shaded areas
    for i in range(len(imu_mean)):
        x, y, z = get_3d_ellipse(
            imu_mean.iloc[i][['latent_0', 'latent_1', 'latent_2']].values,
            imu_std.iloc[i][['latent_0', 'latent_1', 'latent_2']].values
        )
        ax1.plot_surface(x, y, z, alpha=imu_omc_alpha)

        # Update axis limits
        x_min, x_max = min(x_min, x.min()), max(x_max, x.max())
        y_min, y_max = min(y_min, y.min()), max(y_max, y.max())
        z_min, z_max = min(z_min, z.min()), max(z_max, z.max())

    # Plot OMC data points and shaded areas
    for i in range(len(omc_mean)):
        x, y, z = get_3d_ellipse(
            omc_mean.iloc[i][['latent_0', 'latent_1', 'latent_2']].values,
            omc_std.iloc[i][['latent_0', 'latent_1', 'latent_2']].values
        )
        ax1.plot_surface(x, y, z, alpha=imu_omc_alpha)

        # Update axis limits
        x_min, x_max = min(x_min, x.min()), max(x_max, x.max())
        y_min, y_max = min(y_min, y.min()), max(y_max, y.max())
        z_min, z_max = min(z_min, z.min()), max(z_max, z.max())

    # Connect IMU and OMC data points across speeds
    for speed in imu_mean['speed']:
        imu_point = imu_mean[imu_mean['speed'] == speed][['latent_0', 'latent_1', 'latent_2']].values
        omc_point = omc_mean[omc_mean['speed'] == speed][['latent_0', 'latent_1', 'latent_2']].values
        if imu_point.size > 0 and omc_point.size > 0:
            ax1.plot(
                [imu_point[0][0], omc_point[0][0]],
                [imu_point[0][1], omc_point[0][1]],
                [imu_point[0][2], omc_point[0][2]],
                linestyle='--', color='gray', alpha=0.8
            )

# Add axis labels and title for the first figure
ax1.set_xlabel("Latent 0")
ax1.set_ylabel("Latent 1")
ax1.set_zlabel("Latent 2")
ax1.set_title("3D Latent Feature", fontsize=16)

# Save the first figure
save_path_1 = r"C:\Users\gib445\surfdrive - Larsen, A.G. (Aske Gye)@surfdrive.surf.nl\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Figures\3D_latent_plot1.svg"
fig1.savefig(save_path_1, format='svg', bbox_inches='tight')

# Set consistent axis limits
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.set_zlim(z_min, z_max)

# 2nd Figure: Plot for Only Healthy Data
fig2 = plt.figure(figsize=(12, 8))
ax2 = fig2.add_subplot(111, projection='3d')

# Add healthy data blobs in light grey
ax2.scatter(raw_x_test_latent_2d_array['latent_0'], raw_x_test_latent_2d_array['latent_1'], raw_x_test_latent_2d_array['latent_2'],
            c='lightgrey', s=10, alpha=healthy_alpha, label='Validation data')

# Add axis labels and title for the second figure
ax2.set_xlabel("Latent 0")
ax2.set_ylabel("Latent 1")
ax2.set_zlabel("Latent 2")
ax2.set_title("3D Latent Feature", fontsize=16)

# Apply consistent axis limits
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_zlim(z_min, z_max)

# Save the second figure
save_path_2 = r"C:\Users\gib445\surfdrive - Larsen, A.G. (Aske Gye)@surfdrive.surf.nl\PHD Aske\Paper Ideas\OMC vs IMU vs Video\Figures\3D_latent_plot2.svg"
fig2.savefig(save_path_2, format='svg', bbox_inches='tight')

# Show the plots
plt.tight_layout()
plt.show()


#%%

def plot_statistical_parametric_mapping(data1, data2, column_names, plot_name):
    """
    Plot Statistical Parametric Mapping for each joint angle in a 6x6 grid.

    Parameters:
        data1 (ndarray): First dataset (e.g., data_IMU) of shape (samples, timepoints, angles).
        data2 (ndarray): Second dataset (e.g., IMU_reconstructed) of the same shape as data1.
        column_names (list): List of joint angle names corresponding to the third dimension of the datasets.
        plot_name (str): Name prefix for the plots.
    """
    # Define subplot grid size (6 rows, 6 columns)
    n_rows = 6
    n_cols = 6

    # Determine the total number of subplots (it should match the number of angles)
    num_plots = len(column_names)

    # Create a figure with a 6x6 grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18))
    axes = axes.flatten()  # Flatten to access each subplot in a 1D array

    # Iterate over the angles and plot each in the appropriate subplot
    for idx, angle in enumerate(column_names):
        if idx >= len(axes):  # Exit if there are no more subplots to plot
            break

        ax1 = axes[2 * idx]  # Mean plot (top row)
        ax2 = axes[2 * idx + 1]  # t-values plot (bottom row)

        # Extract data for the current angle
        angle_data1 = data1[:, :, idx]
        angle_data2 = data2[:, :, idx]

        # Compute mean and standard deviation
        mean1 = np.mean(angle_data1, axis=0)
        mean2 = np.mean(angle_data2, axis=0)
        std1 = np.std(angle_data1, axis=0)
        std2 = np.std(angle_data2, axis=0)

        # Paired t-test across samples for each subject
        t_values, p_values = ttest_rel(angle_data1, angle_data2, axis=0)

        # Plot mean values with line plots
        ax1.plot(np.arange(data1.shape[1]), mean1, label=f'{angle}', color='blue')
        ax1.plot(np.arange(data1.shape[1]), mean2, label=f'{angle}', color='orange')
        ax1.fill_between(np.arange(data1.shape[1]), mean1 - std1, mean1 + std1, color='blue', alpha=0.2)
        ax1.fill_between(np.arange(data1.shape[1]), mean2 - std2, mean2 + std2, color='orange', alpha=0.2)

        # Check for non-overlapping standard deviation and add shading if necessary
        for j in range(data1.shape[1]):
            if not (mean1[j] - std1[j] <= mean2[j] + std2[j] and mean1[j] + std1[j] >= mean2[j] - std2[j]):
                # If standard deviations do not overlap, shade in light red
                ax1.axvspan(j, j, color='red', alpha=0.2)  # vertical shading at that sample

        # Add labels and legend to mean plot
        ax1.set_title(f'{angle} - Mean Comparison')
        ax1.set_ylabel('Angle')
        ax1.legend()

        # Plot t-values with line plots
        ax2.plot(np.arange(data1.shape[1]), t_values, label='t-values', color='blue')
        ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)

        # Add labels and legend to t-values plot
        ax2.set_title(f'{angle} - t-values')
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('t-value')
        ax2.legend()

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(f'{plot_name}', y=1.02, fontsize=16)
    plt.show()

# Example usage
plot_statistical_parametric_mapping(data_IMU, imu_reconstructed, columns_names, "SPM IMU")
plot_statistical_parametric_mapping(data_OMC, omc_reconstructed, columns_names, "SPM OMC")


#%%
# Function to add 'Other' category points
# Global font size variables
TITLE_FONTSIZE = 26
LEGEND_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 16
POINT_SIZE = 70
HORIZONTAL_OFFSET = -1  # Horizontal offset for the plot
LABEL_PAD = 10  # Padding for axis labels

def generate_color_map(custom_labels, color_order):
    labels = list(custom_labels.keys())
    color_map = {label: color_order[i] for i, label in enumerate(labels)}
    return color_map

def plot_latent_space_3D(cvae, X_test, y_test, latent_features, custom_labels=None, label_name="Fall Risk", 
                         plot_title=None):
    if latent_features != 3:
        print("Latent features are not 3. Skipping 3D plot.")
        return

    # Set the default color order
    default_color_order = ['blue', 'red', 'green', 'yellow']
    
    # Generate the color map for the labels in custom_labels
    if custom_labels is None:
        custom_labels = {label: f'{label_name} {label}' for label in np.unique(y_test)}
    color_map = generate_color_map(custom_labels, default_color_order)

    # Set all unspecified labels to grey
    color_map_default = {label: 'grey' for label in np.unique(y_test)}
    color_map_default.update(color_map)  # Overwrite with specified colors

    encoded_points = []
    labels = []

    for i in range(len(X_test)):
        # Encode each test sample to get its latent space representation
        visualData = np.expand_dims(X_test[i], axis=0)
        mean, logvar = cvae.encode(visualData)
        latent_encoding = cvae.reparameterize(mean, logvar)
        encoded_points.append(latent_encoding[0].numpy())
        labels.append(y_test[i])

    # Convert to numpy array
    encoded_points = np.array(encoded_points)
    labels = np.array(labels)

    # Prepare the dataframe for easy handling
    df = pd.DataFrame(encoded_points, columns=['Latent Feature 1', 'Latent Feature 2', 'Latent Feature 3'])
    df['Label'] = labels

    # Apply color mapping
    df['Color'] = df['Label'].map(lambda x: color_map_default[x])

    # Plot in 3D
    fig = plt.figure(figsize=(12, 16))  # Increase plot size
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['Latent Feature 1'], df['Latent Feature 2'], df['Latent Feature 3'], 
                         c=df['Color'], edgecolors='white', s=POINT_SIZE)

    # Encode healthy other_data samples
    # healthy_indices = np.where(y_adapted == 1)[0]
    # healthy_data = other_data[healthy_indices]
    healthy_data = other_data

    # Encode the healthy other_data samples
    encoded_healthy_data = []
    for i in range(len(healthy_data)):
        visualData = np.expand_dims(healthy_data[i], axis=0)
        mean, logvar = cvae.encode(visualData)
        latent_encoding = cvae.reparameterize(mean, logvar)
        encoded_healthy_data.append(latent_encoding[0].numpy())

    # Convert to numpy array
    encoded_healthy_data = np.array(encoded_healthy_data)

    # Add the healthy data points to the plot
    ax.scatter(encoded_healthy_data[:, 0], encoded_healthy_data[:, 1], encoded_healthy_data[:, 2],
               c='grey', edgecolors='white', s=POINT_SIZE, label='Others')

    # Labeling the axes with padding
    ax.set_xlabel('Latent Feature 1', fontsize=AXIS_LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel('Latent Feature 2', fontsize=AXIS_LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_zlabel('Latent Feature 3', fontsize=AXIS_LABEL_FONTSIZE, labelpad=LABEL_PAD)

    # Auto-scale the axes to fit the data
    ax.auto_scale_xyz(
        [df['Latent Feature 1'].min(), df['Latent Feature 1'].max()],
        [df['Latent Feature 2'].min(), df['Latent Feature 2'].max()],
        [df['Latent Feature 3'].min(), df['Latent Feature 3'].max()]
    )

    # Set custom plot title if provided
    if plot_title is not None:
        plt.title(plot_title, fontsize=TITLE_FONTSIZE)
    else:
        plt.title(f'3D Latent Space Representation ({label_name})', fontsize=TITLE_FONTSIZE)

    # Create a custom legend
    legend_handles = []
    for label, custom_label in custom_labels.items():
        color = color_map_default[label]
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=custom_label,
                                         markerfacecolor=color, markersize=10))

    # Include "Others" category if necessary
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Others',
                                     markerfacecolor='grey', markersize=10))

    if legend_handles:  # Only show legend if there are labels to show
        ax.legend(handles=legend_handles, loc="upper right", fontsize=LEGEND_FONTSIZE)

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

    # Adjust subplot to accommodate horizontal offset
    plt.subplots_adjust(left=HORIZONTAL_OFFSET, right=0.9, 
                        top=0.9, bottom=0.1)

    plt.show()

# Example usage with the provided custom labels
custom_labels_participant_id = {1: 'ID 1', 5: 'ID 5', 9: 'ID 9', 11: 'ID 11'}
plot_latent_space_3D(cvae, test_data, participant_id, latent_features=3, custom_labels=custom_labels_participant_id, label_name="Participant ID", 
                      plot_title="Latent Space - Participant IDs")


#%%

# # Initialize arrays to store correlations for both training and testing data
# correlation_train = []
# correlation_test = []

# # Calculate correlations for training data
# for indx in range(len(X_train)):
#     tempDimensiontrain = np.expand_dims(X_train[indx, :, :], axis=0)
#     predictedtrain = cvae.decode(cvae.reparameterize(*cvae.encode(tempDimensiontrain))).numpy()
#     predictedtrain = predictedtrain.reshape(window_length, number_of_columns)

#     correlationtemp = []
#     for indx2 in range(number_of_columns):
#         correlationtemp.append(np.corrcoef(predictedtrain[:, indx2], X_train[indx, :, indx2])[0, 1])
#     correlation_train.append(correlationtemp)

# # Calculate correlations for testing data
# for indx in range(len(X_test)):
#     tempDimensiontest = np.expand_dims(X_test[indx, :, :], axis=0)
#     predictedtest = cvae.decode(cvae.reparameterize(*cvae.encode(tempDimensiontest))).numpy()
#     predictedtest = predictedtest.reshape(window_length, number_of_columns)

#     correlationtemp = []
#     for indx2 in range(number_of_columns):
#         correlationtemp.append(np.corrcoef(predictedtest[:, indx2], X_test[indx, :, indx2])[0, 1])
#     correlation_test.append(correlationtemp)

# # Convert lists to numpy arrays for easier manipulation
# correlation_train = np.array(correlation_train)
# correlation_test = np.array(correlation_test)

# # Create boxplots for correlations
# boxplottraindata = [correlation_train[:, i] for i in range(number_of_columns)]
# boxplottestdata = [correlation_test[:, i] for i in range(number_of_columns)]

# # Plot the correlation boxplots for training and testing datasets
# fig = plt.figure(24, figsize=(10, 8))

# ax1 = fig.add_subplot(211)
# ax1.title.set_text('Correlation boxplot: training dataset')
# bplot1 = ax1.boxplot(boxplottraindata, patch_artist=True, showfliers=False)

# ax2 = fig.add_subplot(212)
# ax2.title.set_text('Correlation boxplot: testing dataset')
# bplot2 = ax2.boxplot(boxplottestdata, patch_artist=True, showfliers=False)

# # Customize colors
# colors = ['pink', 'lightblue', 'lightgreen'] * 6  # Repeated colors for more columns
# for bplot in (bplot1, bplot2):
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)

# plt.show()

#%%

# # Initialize lists to collect errors
# error_train = []
# Norerror_train = []

# # Calculate RMSE and nRMSE for each training instance
# for indx in range(len(X_train)):
#     tempDimensiontrain = np.expand_dims(X_train[indx, :, :], axis=0)
#     predictedtrain = cvae.decode(cvae.reparameterize(*cvae.encode(tempDimensiontrain))).numpy()
#     predictedtrain = predictedtrain.reshape(window_length, number_of_columns)
    
#     errortemp = []
#     Norerrortemp = []
    
#     for indx2 in range(number_of_columns):
#         # Calculate RMSE
#         rmse_value = rmse(predictedtrain[:, indx2], X_train[indx, :, indx2])
#         errortemp.append(rmse_value)
        
#         # Calculate nRMSE (normalized by the range)
#         nrmse_value = rmse_value / (np.max(X_train[indx, :, indx2]) - np.min(X_train[indx, :, indx2]))
#         Norerrortemp.append(nrmse_value)
    
#     error_train.append(errortemp)
#     Norerror_train.append(Norerrortemp)

# # Convert lists to numpy arrays for further processing
# error_train = np.array(error_train)
# Norerror_train = np.array(Norerror_train)

# # Prepare data for boxplots
# boxplottraindata_RMSE = [error_train[:, i] for i in range(number_of_columns)]
# boxplottraindata_nRMSE = [Norerror_train[:, i] for i in range(number_of_columns)]

# # Set y-axis limits
# ylim = 60
# ymin = 0

# # Create figure
# fig = plt.figure(18, figsize=(14, 10))

# # RMSE boxplot: training dataset
# ax1 = fig.add_subplot(221)
# ax1.title.set_text('RMSE boxplot: training dataset')
# bplot1 = ax1.boxplot(boxplottraindata_RMSE, patch_artist=True, showfliers=False)
# ax1.set_ylim(ymin, ylim)
# ax1.set_ylabel('degrees')

# # Normalized RMSE boxplot: training dataset
# ax2 = fig.add_subplot(222)
# ax2.title.set_text('Normalized RMSE boxplot: training dataset')
# bplot2 = ax2.boxplot(boxplottraindata_nRMSE, patch_artist=True, showfliers=False)
# ax2.set_ylim(ymin, ylim)
# ax2.set_ylabel('percentage')

# # Custom colors for the boxplots
# colors = ['pink', 'lightblue', 'lightgreen', 'pink', 'lightblue', 'lightgreen']

# for bplot in (bplot1, bplot2):
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)

# plt.show()


#%%
# Define font size variables
LEGEND_FONTSIZE = 50
TITLE_FONTSIZE = 50
AXIS_LABEL_FONTSIZE = 35
TICK_LABEL_FONTSIZE = 25

# Define plot layout variables
FIGURE_WIDTH = 70          # Width of the entire figure
FIGURE_HEIGHT = 40         # Height of the entire figure
SUBPLOT_WIDTH = 0.13       # Width of each subplot
SUBPLOT_HEIGHT = 0.35    # Height of each subplot
HORIZONTAL_SPACING = 0.01 # Horizontal space between subplots
VERTICAL_SPACING = -0.01    # Vertical space between subplots
LEGEND_POSITION = (0.33, -0.1)  # Position of the global legend (bottom)

def plot_latent_space_3D_for_all_participants(cvae, X_test, week_numbers, participant_id, latent_features=3, colored_point_size=400):
    if latent_features != 3:
        print("Latent features are not 3. Skipping 3D plot.")
        return

    # Color map using different shades of blue
    color_map_weeks = {1: 'lightblue', 9: 'skyblue', 17: 'dodgerblue', 26: 'blue'}

    unique_participants = np.unique(participant_id)
    num_participants = len(unique_participants)
    
    # Create a figure with a large enough size to accommodate all subplots and legend
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    rows = 3
    cols = 5

    # Create a list to hold legend handles for the global legend
    legend_handles = []
    for week, color in color_map_weeks.items():
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label=f'Week {week}', 
                                     markerfacecolor=color, markersize=40))

    for i, chosen_participant_id in enumerate(unique_participants):
        row = i // cols
        col = i % cols

        # Calculate position of the subplot
        x_position = col * (SUBPLOT_WIDTH + HORIZONTAL_SPACING)
        y_position = 1 - (row * (SUBPLOT_HEIGHT + VERTICAL_SPACING) + SUBPLOT_HEIGHT)

        # Filter the data for the chosen participant
        participant_mask = (participant_id == chosen_participant_id)
        chosen_data = X_test[participant_mask]
        chosen_weeks = week_numbers[participant_mask]
        
        if len(chosen_data) == 0:
            continue
        
        encoded_points = []
        colors = []
        sizes = []

        # Encoding all the data points for the chosen participant
        for i in range(len(chosen_data)):
            visualData = np.expand_dims(chosen_data[i], axis=0)
            mean, logvar = cvae.encode(visualData)
            latent_encoding = cvae.reparameterize(mean, logvar)
            encoded_points.append(latent_encoding[0].numpy())
            
            colors.append(color_map_weeks.get(chosen_weeks[i], 'blue'))
            sizes.append(colored_point_size)

        # Convert to numpy array
        encoded_points = np.array(encoded_points)

        # Prepare the dataframe for easy handling
        df = pd.DataFrame(encoded_points, columns=['Latent Feature 1', 'Latent Feature 2', 'Latent Feature 3'])
        df['Color'] = colors
        df['Size'] = sizes

        # Add subplot for the current participant
        ax = fig.add_axes([x_position, y_position, SUBPLOT_WIDTH, SUBPLOT_HEIGHT], projection='3d')
        scatter = ax.scatter(df['Latent Feature 1'], df['Latent Feature 2'], df['Latent Feature 3'], 
                             c=df['Color'], edgecolors='white', s=df['Size'])

        # Calculate mean points for each week
        mean_points = []
        mean_colors = []
        for week in sorted(set(chosen_weeks)):
            week_mask = (chosen_weeks == week)
            mean_point = df[week_mask][['Latent Feature 1', 'Latent Feature 2', 'Latent Feature 3']].mean().values
            mean_points.append(mean_point)
            mean_colors.append(color_map_weeks.get(week, 'blue'))

        # Connect the mean points with a line that transitions between colors
        mean_points = np.array(mean_points)
        for i in range(len(mean_points) - 1):
            start_color = mean_colors[i]
            end_color = mean_colors[i + 1]
            
            # Define a custom colormap that transitions between start and end colors
            cmap = LinearSegmentedColormap.from_list('grad_cmap', [start_color, end_color])
            for j in np.linspace(0, 1, 100):  # Use 100 small segments for smooth transition
                color = cmap(j)
                start_point = mean_points[i] * (1 - j) + mean_points[i + 1] * j
                end_point = mean_points[i] * (1 - (j + 0.01)) + mean_points[i + 1] * (j + 0.01)
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]],
                        color=color, linewidth=7)

        # Labeling the axes with padding
        ax.set_xlabel('Latent Feature 1', fontsize=AXIS_LABEL_FONTSIZE, labelpad=20)
        ax.set_ylabel('Latent Feature 2', fontsize=AXIS_LABEL_FONTSIZE, labelpad=20)
        ax.set_zlabel('Latent Feature 3', fontsize=AXIS_LABEL_FONTSIZE, labelpad=20)

        # Set custom plot title with the participant ID
        ax.set_title(f'Participant ID {int(chosen_participant_id)}', fontsize=TITLE_FONTSIZE)

        # Set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis='both', which='minor', labelsize=TICK_LABEL_FONTSIZE)

    # Adjust spacing around subplots to accommodate the global legend
    plt.subplots_adjust(left=0.0, right=1, top=0.1, bottom=0.0, wspace=HORIZONTAL_SPACING, hspace=VERTICAL_SPACING)

    # Add a single global legend at the bottom
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(LEGEND_POSITION[0], LEGEND_POSITION[1]), ncol=4, fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()
    plt.show()

# Example usage
plot_latent_space_3D_for_all_participants(cvae, test_data, week_numbers, participant_id, latent_features=3, colored_point_size=400)

#%%

# Obtain latent features for training and test data
def get_latent_features(model, data):
    mean, logvar = model.encode(data)
    z = model.reparameterize(mean, logvar)
    return z, mean, logvar

# Calculate latent features for the training data
latent_features, means, log_vars = get_latent_features(cvae, X_train)

# Calculate the variance of latent features using distance to cluster means
def calculate_latent_distance_variance(latent_features, participant_id, week_numbers):
    # Convert latent features to numpy array
    latent_features_np = latent_features.numpy()
    
    # Dictionary to hold variances
    variances = {}
    
    # Get unique participants and week numbers
    unique_participants = np.unique(participant_id)
    unique_weeks = np.unique(week_numbers)
    
    for participant in unique_participants:
        participant_indices = np.where(participant_id == participant)[0]
        
        # Compute latent feature variances for this participant
        participant_latent_features = latent_features_np[participant_indices]
        variances[participant] = {}
        
        for week in unique_weeks:
            week_indices = np.where(week_numbers[participant_indices] == week)[0]
            week_latent_features = participant_latent_features[week_indices]
            
            if week_latent_features.shape[0] > 1:
                # Calculate mean vector for this participant-week combination
                mean_vector = np.mean(week_latent_features, axis=0)
                
                # Calculate distances from each point to the mean vector
                distances = np.linalg.norm(week_latent_features - mean_vector, axis=1)
                
                # Variance is the mean of squared distances
                distance_variance = np.mean(distances ** 2)
                variances[participant][week] = distance_variance
            else:
                variances[participant][week] = np.nan  # Handle case with insufficient data
    
    return variances

# Use the updated function to calculate the variances
variances = calculate_latent_distance_variance(latent_features, participant_id, week_numbers)


# Prepare data for a single plot
def prepare_data_for_single_plot(variances, unique_weeks):
    participants = sorted(variances.keys())
    num_participants = len(participants)
    num_weeks = len(unique_weeks)
    
    # Create arrays to hold x, y, and sizes for the plot
    x_values = []
    y_values = []
    sizes = []

    for i, participant in enumerate(participants):
        for week in unique_weeks:
            x_values.append(week)
            y_values.append(i)
            sizes.append(variances[participant].get(week, np.nan))  # Handle missing data

    return np.array(x_values), np.array(y_values), np.array(sizes), participants

# Calculate unique weeks
unique_weeks = np.unique(week_numbers)
# Prepare data for plotting
x_values, y_values, sizes, participants = prepare_data_for_single_plot(variances, unique_weeks)


def plot_latent_variance(x_values, y_values, sizes, participants, unique_weeks,
                         title_fontsize=16, xlabel_fontsize=14, ylabel_fontsize=14, 
                         tick_fontsize=12, colorbar_fontsize=12):
    # Create figure and axis
    fig = plt.figure(figsize=(12, 24))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[30, 1], hspace=0.14)
    ax = plt.subplot(gs[0])
    cax = plt.subplot(gs[1])
    
    # Create scatter plot with adjusted blob size
    scatter = ax.scatter(x_values, y_values, s=sizes * 7, alpha=0.6, c=sizes, cmap='viridis', edgecolor='k')

    # Define mapping for x-axis values
    x_tick_mapping = {0: 1, 1: 9, 2: 17, 3: 26}
    
    # Set x-axis ticks and labels
    x_ticks = sorted(x_tick_mapping.keys())
    x_tick_labels = [x_tick_mapping[tick] for tick in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=tick_fontsize)

    # Set y-axis ticks to be integers starting from 1
    ax.set_yticks(np.arange(len(participants)))
    ax.set_yticklabels(np.arange(1, len(participants) + 1), fontsize=tick_fontsize)

    # Set axis labels and title
    ax.set_xlabel('Week Number', fontsize=xlabel_fontsize)
    ax.set_ylabel('Participant', fontsize=ylabel_fontsize)
    ax.set_title('Variance of Latent Features by Participant and Week', fontsize=title_fontsize)

    # Add color bar with manual positioning
    cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')
    cbar.set_label('Variance', fontsize=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)

    # Adjust x-axis limits to reduce spacing between ticks
    ax.set_xlim(min(x_ticks) - 0.3, max(x_ticks) + 0.3)
    ax.set_ylim(-1, len(participants))  # Increase vertical space

    # Remove grid
    ax.grid(False)  # Remove grid lines

    plt.show()

# Example usage with custom font sizes
plot_latent_variance(
    x_values, y_values, sizes, participants, unique_weeks,
    title_fontsize=28, xlabel_fontsize=25, ylabel_fontsize=25,
    tick_fontsize=22, colorbar_fontsize=25
)

#%%

# Modified plotting function to include subplots for different views
def plot_latent_space_3D(cvae, X_test, y_test, latent_features, custom_labels=None, label_name="Fall Risk", 
                         plot_title=None, test_data=None):
    if latent_features != 3:
        print("Latent features are not 3. Skipping 3D plot.")
        return

    # Set the default color order
    default_color_order = ['blue', 'red', 'green', 'yellow', 'purple']
    
    # Generate the color map for the labels in custom_labels
    if custom_labels is None:
        custom_labels = {label: f'{label_name} {label}' for label in np.unique(y_test)}
    color_map = generate_color_map(custom_labels, default_color_order)

    # Set all unspecified labels to grey
    color_map_default = {label: 'grey' for label in np.unique(y_test)}
    color_map_default.update(color_map)  # Overwrite with specified colors

    encoded_points = []
    labels = []

    for i in range(len(X_test)):
        # Encode each test sample to get its latent space representation
        visualData = np.expand_dims(X_test[i], axis=0)
        mean, logvar = cvae.encode(visualData)
        latent_encoding = cvae.reparameterize(mean, logvar)
        encoded_points.append(latent_encoding[0].numpy())
        labels.append(y_test[i])

    # Encode test_data if provided
    encoded_test_data = None
    if test_data is not None:
        encoded_test_data = []
        for i in range(len(test_data)):
            visualData = np.expand_dims(test_data[i], axis=0)
            mean, logvar = cvae.encode(visualData)
            latent_encoding = cvae.reparameterize(mean, logvar)
            encoded_test_data.append(latent_encoding[0].numpy())
        encoded_test_data = np.array(encoded_test_data)

    # Convert to numpy array
    encoded_points = np.array(encoded_points)
    labels = np.array(labels)

    # Prepare the dataframe for easy handling
    df = pd.DataFrame(encoded_points, columns=['Latent Feature 1', 'Latent Feature 2', 'Latent Feature 3'])
    df['Label'] = labels

    # Apply color mapping
    df['Color'] = df['Label'].map(lambda x: color_map_default[x])

    # Prepare subplots
    fig = plt.figure(figsize=(18, 12))  # Adjust figure size
    view_angles = [(30, 60), (30, 120), (30, 180), (30, 240), (30, 300), (60, 30)]

    for idx, angle in enumerate(view_angles):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        scatter = ax.scatter(df['Latent Feature 1'], df['Latent Feature 2'], df['Latent Feature 3'], 
                             c=df['Color'], edgecolors='white', s=POINT_SIZE, label='Test Data')

        # Plot test_data if available
        if encoded_test_data is not None:
            ax.scatter(encoded_test_data[:, 0], encoded_test_data[:, 1], encoded_test_data[:, 2], 
                       c='orange', edgecolors='white', s=POINT_SIZE, label='New Data')

        ax.view_init(angle[0], angle[1])

        # Labeling the axes with padding
        ax.set_xlabel('Latent Feature 1', fontsize=AXIS_LABEL_FONTSIZE, labelpad=LABEL_PAD)
        ax.set_ylabel('Latent Feature 2', fontsize=AXIS_LABEL_FONTSIZE, labelpad=LABEL_PAD)
        ax.set_zlabel('Latent Feature 3', fontsize=AXIS_LABEL_FONTSIZE, labelpad=LABEL_PAD)

        # Set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)

        # Set title for each subplot based on the viewing angle
        ax.set_title(f'View Angle ({angle[0]}, {angle[1]})', fontsize=TITLE_FONTSIZE)

    # Create a custom legend
    legend_handles = []
    for label, custom_label in custom_labels.items():
        color = color_map_default[label]
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label=custom_label,
                                     markerfacecolor=color, markersize=10))

    # Include "Others" category if necessary
    if len(np.unique(labels)) > len(custom_labels):
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label='Others',
                                     markerfacecolor='grey', markersize=10))

    # Add New Data to legend
    legend_handles.append(Line2D([0], [0], marker='o', color='w', label='New Data',
                                 markerfacecolor='orange', markersize=10))

    if legend_handles:  # Only show legend if there are labels to show
        fig.legend(handles=legend_handles, loc="upper right", fontsize=LEGEND_FONTSIZE)

    # Adjust subplot layout
    plt.tight_layout()
    plt.show()

# Example usage, assuming `cvae`, `X_test`, `y_test`, and `test_data` are defined
plot_latent_space_3D(cvae, X_test, y_test, latent_features=3, custom_labels=custom_labels_classes, label_name="Fall Risk", 
                     plot_title="Latent Space - Fall Risk Levels", test_data=test_data)

#%%

# Example column names for the plots
column_names = [
    'LKneeAngles_X', 'LKneeAngles_Y', 'LKneeAngles_Z',
    'LHipAngles_X', 'LHipAngles_Y', 'LHipAngles_Z',
    'LAnkleAngles_X', 'LAnkleAngles_Y', 'LAnkleAngles_Z',
    'RKneeAngles_X', 'RKneeAngles_Y', 'RKneeAngles_Z',
    'RHipAngles_X', 'RHipAngles_Y', 'RHipAngles_Z',
    'RAnkleAngles_X', 'RAnkleAngles_Y', 'RAnkleAngles_Z'
]

# Custom labels for weeks
custom_labels_weeks = {0: '01', 1: '09', 2: '17', 3: '26'}
week_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}  # Define colors for each week

def plot_individual_participant_data(other_data, test_data, y_adapted, participant_id, week_numbers, column_names):
    # Filter other_data using y_adapted, only keep rows where y_adapted == 1 (healthy people)
    healthy_data = other_data[y_adapted.flatten() == 1]

    # Calculate the standard deviation and mean across the healthy samples for other_data
    std_healthy = np.std(healthy_data, axis=0)  # Shape: (200, 18)
    mean_healthy = np.mean(healthy_data, axis=0)  # Shape: (200, 18)

    # Get unique participant IDs
    unique_participants = np.unique(participant_id)

    # Loop over each unique participant
    for participant in unique_participants:
        # Filter test_data and week_numbers for the current participant
        participant_data = test_data[participant_id.flatten() == participant]
        participant_weeks = week_numbers[participant_id.flatten() == participant]
        
        if participant_data.size == 0:
            continue

        # Create a new figure for the current participant
        fig, axes = plt.subplots(6, 3, figsize=(18, 20))
        
        # Set a larger and higher title
        fig.suptitle(f'Participant {int(participant)}', fontsize=30, y=0.95)

        # Initialize handles and labels for the legend
        handles = []
        labels = []

        # Loop over each column and plot in the corresponding subplot
        for i, column_name in enumerate(column_names):
            row, col = divmod(i, 3)
            ax = axes[row, col]

            # Plot the shaded area for standard deviation centered around the mean of healthy_data
            ax.fill_between(
                range(std_healthy.shape[0]),
                mean_healthy[:, i] - std_healthy[:, i],
                mean_healthy[:, i] + std_healthy[:, i],
                color='lightgrey', alpha=0.7, label='Healthy STD'
            )

            # Plot the mean of test_data for each week number as a separate line
            for week_num in np.unique(participant_weeks):
                week_data = participant_data[participant_weeks.flatten() == week_num]
                if week_data.size > 0:
                    mean_week = np.mean(week_data, axis=0)
                    line, = ax.plot(mean_week[:, i], color=week_colors[week_num], label=f'Week {custom_labels_weeks[week_num]}')
                    if line.get_label() not in labels:
                        handles.append(line)
                        labels.append(line.get_label())

            # Set titles and labels
            ax.set_title(column_name)
            ax.set_xlabel('Time Point')
            ax.set_ylabel('Value')
            ax.grid(True)

        # Add a legend to the top right of the entire plot
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.9, 1), bbox_transform=fig.transFigure, fontsize=12)

        # Adjust layout to accommodate the legend at the top right
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Leave space for the legend and larger title
        plt.savefig(f'participant_{int(participant)}_latent_features.svg', format='svg')

        plt.show()
        

plot_individual_participant_data(other_data, test_data, y_adapted, participant_id, week_numbers, column_names)
