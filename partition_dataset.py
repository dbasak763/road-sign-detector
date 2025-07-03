import os
import shutil
from sklearn.model_selection import train_test_split

def partition_dataset():
    # Set up paths
    project_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(project_dir, 'Road_Sign_Dataset')
    images_dir = os.path.join(dataset_dir, 'images')
    annotations_dir = os.path.join(dataset_dir, 'annotations')

    # Read images and annotations
    images = [os.path.join(images_dir, x) for x in os.listdir(images_dir) if x.endswith(('.png', '.jpg', '.jpeg'))]
    annotations = [os.path.join(annotations_dir, os.path.splitext(os.path.basename(x))[0] + '.txt') for x in images]
    
    images.sort()
    annotations.sort()

    # Split the dataset
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size=0.2, random_state=1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size=0.5, random_state=1)

    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(annotations_dir, split), exist_ok=True)

    # Utility function to move files
    def move_files_to_folder(list_of_files, destination_folder):
        for f in list_of_files:
            try:
                shutil.move(f, destination_folder)
            except FileNotFoundError:
                print(f"File not found: {f}. Skipping.")
            except Exception as e:
                print(f"Error moving file {f}: {e}")

    # Move the files
    move_files_to_folder(train_images, os.path.join(images_dir, 'train'))
    move_files_to_folder(val_images, os.path.join(images_dir, 'val'))
    move_files_to_folder(test_images, os.path.join(images_dir, 'test'))
    move_files_to_folder(train_annotations, os.path.join(annotations_dir, 'train'))
    move_files_to_folder(val_annotations, os.path.join(annotations_dir, 'val'))
    move_files_to_folder(test_annotations, os.path.join(annotations_dir, 'test'))

    # Rename annotations folder to labels
    labels_dir = os.path.join(dataset_dir, 'labels')
    if os.path.exists(labels_dir):
        print("'labels' directory already exists. Skipping rename.")
    else:
        os.rename(annotations_dir, labels_dir)
        print("Renamed 'annotations' directory to 'labels'.")

    print("Dataset partitioning complete.")

if __name__ == "__main__":
    partition_dataset()
