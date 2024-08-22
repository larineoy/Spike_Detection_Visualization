import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Path to the image file
image_path = '/Users/larineouyang/GitHub/Spike_Detection_Visualization/spike_detection_result/m_annotation/spike_detection_4570178-1_20220218_143504.png'

# Load the image using OpenCV
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load a template of the "m" annotation (you need to create this template)
# You can create this template by manually cropping a small portion of the image where "m" appears.
# For simplicity, let's assume you have this template saved as "m_template.png".
template_path = '/Users/larineouyang/GitHub/Spike_Detection_Visualization/spike_detection_result/m_annotation/m_template.png'
template = cv2.imread(template_path, 0)  # Load as grayscale

# Perform template matching to find "m" annotations
result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)

# Define a threshold to detect the template
threshold = 0.8  # Adjust based on the match quality
locations = np.where(result >= threshold)

# Directory to save the cropped images
save_directory = "/Users/larineouyang/GitHub/Spike_Detection_Visualization/spike_detection_result/m_annotation/m_4570178-1_20220218_143504"
os.makedirs(save_directory, exist_ok=True)

# Loop through detected locations and extract segments
h, w = template.shape  # Height and width of the template

for i, (x, y) in enumerate(zip(locations[1], locations[0]), start=1):
    # Calculate the coordinates for a 10-second window around the "m" annotation
    # Here, we'll assume a fixed region around each detected "m"
    # You may need to adjust these values based on your spectrogram's scale and resolution
    start_x = max(0, x - w)
    start_y = max(0, y - h)
    end_x = x + 2 * w
    end_y = y + 2 * h
    
    # Crop the detected region
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    # Convert the cropped region back to PIL image to save it
    cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    
    # Save the cropped image with the naming convention
    file_name = f"m({i})_4570178-1_20220218_143504.png"
    save_path = os.path.join(save_directory, file_name)
    cropped_pil_image.save(save_path)

    # Optionally, display each cropped image
    plt.imshow(cropped_pil_image)
    plt.title(f"m({i}) Annotation Segment")
    plt.axis('off')
    plt.show()

print("Cropped images saved successfully.")
