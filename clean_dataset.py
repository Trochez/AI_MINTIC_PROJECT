import os

def remove_invalid_files(image_dir, label_dir):
    # List all files in the label directory
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        image_name = os.path.splitext(label_file)[0] + '.jpg'
        image_path = os.path.join(image_dir, image_name)

        with open(label_path, 'r') as file:
            lines = file.readlines()

            # Check if the label file is empty or has more than one line
            if len(lines) == 0 or len(lines) > 1:
                print(f"Removing {image_name} and {label_file}")
                os.remove(label_path)  # Remove the label file
                if os.path.exists(image_path):
                    os.remove(image_path)  # Remove the corresponding image file

# Example usage:
remove_invalid_files('./Rock Paper Scissors SXSW.v14i.yolov8/train/images/', './Rock Paper Scissors SXSW.v14i.yolov8/train/labels/')

