import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Define the paths (use absolute paths)
spike_prob_dir = "/Users/larineouyang/spike_prob_results/spike_prob_results"
spectrogram_dir = "/Users/larineouyang/Downloads/Sleep_EEG_Spectrogram_Figures"
output_dir = "/Users/larineouyang/spike_detection_result"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to process a single pair of spike probability result and spectrogram
def process_and_plot(spike_prob_file, spectrogram_file, output_file):
    # Load spike probability data
    spike_data = np.load(spike_prob_file)
    
    # Extract 'p_spike' and 'Fs'
    if 'p_spike' in spike_data.files and 'Fs' in spike_data.files:
        probability = spike_data['p_spike']
        Fs = spike_data['Fs']
        time = np.arange(len(probability)) / Fs  # Generate time axis
    else:
        print(f"'p_spike' or 'Fs' key not found in {spike_prob_file}")
        return
    
    # Apply the 0.43 cutoff to detect spikes
    cutoff = 0.43
    spikes = probability > cutoff

    # Load the spectrogram image
    spectrogram = Image.open(spectrogram_file)

    # Plot the data with two vertically aligned subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Display the spectrogram
    ax1.imshow(spectrogram, aspect='auto')
    ax1.set_title('Spectrogram with Spike Detection')
    ax1.set_ylabel('Frequency')

    # Plot the spike probability line with detected spikes
    ax2.plot(time, probability, color='red', label='Spike Detection Probability')
    ax2.scatter(time[spikes], probability[spikes], color='blue', label='Detected Spikes', zorder=5)
    ax2.axhline(y=cutoff, color='black', linestyle='--', label='Cutoff = 0.43')
    ax2.set_ylabel('Spike Probability')
    ax2.set_xlabel('Time')
    
    # Combine the legends from both plots
    ax2.legend(loc='upper right')

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Process all files
# for spike_prob_file in os.listdir(spike_prob_dir):
#     if spike_prob_file.endswith('.npz'):
#         # Extract index from spike probability file name
#         index = spike_prob_file.replace('spike_prob_', '').replace('_fil.npz', '')

#         # Construct the corresponding spectrogram file name based on the updated convention
#         spectrogram_file_name = f"{index}_fil_clean-spectrogram.png"
#         spectrogram_file_path = os.path.join(spectrogram_dir, spectrogram_file_name)

#         # Ensure the spectrogram file exists
#         if os.path.exists(spectrogram_file_path):
#             # Construct the full path for input files and output file
#             spike_prob_file_path = os.path.join(spike_prob_dir, spike_prob_file)
#             output_file_path = os.path.join(output_dir, f"spike_detection_{index}.png")

#             # Process and generate the plot
#             process_and_plot(spike_prob_file_path, spectrogram_file_path, output_file_path)
#         else:
#             print(f"Spectrogram file for index {index} not found: {spectrogram_file_name}")

for spike_prob_file in os.listdir(spike_prob_dir):
    if spike_prob_file.endswith('.npz'):
        # Extract the base index by removing 'spike_prob_' and '_fil.npz'
        index = spike_prob_file.replace('spike_prob_', '').replace('_fil.npz', '')
        
        # Construct the correct spectrogram file name
        spectrogram_file_name = f"{index}_fil_clean-spectrogram.png"
        spectrogram_file_path = os.path.join(spectrogram_dir, spectrogram_file_name)

        # Ensure the spectrogram file exists
        if os.path.exists(spectrogram_file_path):
            # Construct the full path for input files and output file
            spike_prob_file_path = os.path.join(spike_prob_dir, spike_prob_file)
            output_file_path = os.path.join(output_dir, f"spike_detection_{index}.png")

            # Process and generate the plot
            process_and_plot(spike_prob_file_path, spectrogram_file_path, output_file_path)
        else:
            print(f"Spectrogram file for index {index} not found: {spectrogram_file_name}")

