import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def test_annotations():
    # Define class mappings
    class_name_to_id_mapping = {
        "trafficlight": 0,
        "stop": 1,
        "speedlimit": 2,
        "crosswalk": 3
    }
    class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

    def plot_bounding_box(image, annotation_list, save_path):
        annotations = np.array(annotation_list)
        w, h = image.size
        
        plotted_image = ImageDraw.Draw(image)
        
        transformed_annotations = np.copy(annotations)
        transformed_annotations[:, [1, 3]] *= w
        transformed_annotations[:, [2, 4]] *= h 
        
        transformed_annotations[:, 1] -= (transformed_annotations[:, 3] / 2)
        transformed_annotations[:, 2] -= (transformed_annotations[:, 4] / 2)
        
        transformed_annotations[:, 3] += transformed_annotations[:, 1]
        transformed_annotations[:, 4] += transformed_annotations[:, 2]
        
        for ann in transformed_annotations:
            obj_cls, x0, y0, x1, y1 = ann
            plotted_image.rectangle(((x0, y0), (x1, y1)), outline="red", width=3)
            plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[int(obj_cls)], fill="red")
        
        plt.figure(figsize=(12, 12))
        plt.imshow(np.array(image))
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Annotation test image saved to: {save_path}")

    # Set paths
    project_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(project_dir, 'Road_Sign_Dataset')
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    images_dir = os.path.join(dataset_dir, 'images')

    # Get a random annotation file
    txt_annotations = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
    if not txt_annotations:
        print("No .txt annotation files found.")
        return
        
    random.seed(0)
    annotation_filename = random.choice(txt_annotations)
    annotation_path = os.path.join(annotations_dir, annotation_filename)

    # Read annotation file
    with open(annotation_path, "r") as file:
        lines = file.read().strip().split('\n')
        if not lines or not lines[0]:
            print(f"Annotation file {annotation_filename} is empty. Cannot process.")
            return
        annotation_list = [line.split(" ") for line in lines]
        annotation_list = [[float(y) for y in x] for x in annotation_list]

    # Find corresponding image file
    image_name_stem = os.path.splitext(annotation_filename)[0]
    possible_extensions = ['.png', '.jpg', '.jpeg']
    image_path = None
    for ext in possible_extensions:
        potential_path = os.path.join(images_dir, image_name_stem + ext)
        if os.path.exists(potential_path):
            image_path = potential_path
            break

    if not image_path:
        print(f"Could not find corresponding image for {annotation_filename}")
        return

    # Load image and plot
    image = Image.open(image_path)
    output_path = os.path.join(project_dir, 'test_annotation_output.png')
    plot_bounding_box(image, annotation_list, output_path)

if __name__ == "__main__":
    test_annotations()
