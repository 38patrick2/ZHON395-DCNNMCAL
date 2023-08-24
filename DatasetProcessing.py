import os
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# Constants
IMAGE_SIZE = 224
DATA_DIR = '/home/vboxuser/Wrok/ZHON/Dataset/train'
OUTPUT_DIR = '/home/vboxuser/Wrok/ZHON/Dataset/VizOutput'

# Function to read .bytes file and return as numpy array
def read_bytes_file(filepath):
    with open(filepath, 'r') as f:
        hex_values = f.read().split()[1:]  # Ignore addresses at the start of each line
        byte_values = [int(val, 16) for val in hex_values if val != '??']  # Convert hex to int and ignore missing values
        return np.array(byte_values, dtype=np.uint8)

# Function to reshape the data to a fixed image size
def reshape_to_image_size(data, image_size):
    if len(data) > image_size * image_size:
        return data[:image_size * image_size].reshape((image_size, image_size))
    else:
        padded_data = np.pad(data, (0, image_size * image_size - len(data)), 'constant', constant_values=0)
        return padded_data.reshape((image_size, image_size))

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming grayscale input
])

# Process .bytes files
for filename in os.listdir(DATA_DIR):
    if filename.endswith('.bytes'):
        filepath = os.path.join(DATA_DIR, filename)
        
        # Read bytes content as a numpy array
        bytes_array = read_bytes_file(filepath)
        
        # Reshape bytes array to grayscale image
        image_array = reshape_to_image_size(bytes_array, IMAGE_SIZE)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_array, 'L')
        
        # Apply transformations
        transformed_image = transform(pil_image)
        
        # Convert back to PIL Image for saving
        save_image = transforms.ToPILImage()(transformed_image)
        
        # Save image
        output_filename = os.path.splitext(filename)[0] + '.png'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        save_image.save(output_path)

print("Processing and saving images complete.")
