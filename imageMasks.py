import cv2
import numpy as np
import os

def red_mask_intensity(image_path, output_dir):
    # Read the original image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded properly
    if image is None:
        print(f"Error: Could not read the image {image_path}.")
        return
    
    # Convert the image to the RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define the lower and upper boundaries for the red color in the RGB space
    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 100, 100])
    
    # Create a mask that identifies the red regions
    mask = cv2.inRange(image_rgb, lower_red, upper_red)
    
    # Exclude regions that are too green or too blue
    green_channel = image_rgb[:, :, 1]
    blue_channel = image_rgb[:, :, 2]
    #exclude_mask = (green_channel > 100) & (blue_channel > 100)
    
    # Apply the exclusion mask to the red mask
    #mask[exclude_mask] = 0
    
    # Convert the red mask to a grayscale image
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Convert the grayscale mask to float32
    gray_mask_float = gray_mask.astype(np.float32)
    
    # Normalize each channel by dividing by 255
    gray_mask_float /= 255.0

    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save the normalized grayscale mask to the output directory
    output_path = os.path.join(output_dir, filename + '.jpg')
    cv2.imwrite(output_path, (gray_mask_float * 255).astype(np.uint8))
    print(f"Grayscale intensity mask saved to {output_path}")

# Process all images in the FOVImages directory
input_dir = 'FOVImages'
output_dir = 'MaskImages'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        red_mask_intensity(image_path, output_dir)
