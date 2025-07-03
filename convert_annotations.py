import os
import sys
import xml.etree.ElementTree as ET
from tqdm import tqdm

def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()

    info_dict = {}
    info_dict['bboxes'] = []

    for elem in root:
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                if subelem.text is not None:
                    image_size.append(int(subelem.text))
            if len(image_size) == 3:
                 info_dict['image_size'] = tuple(image_size)
            else:
                 info_dict['image_size'] = None
        
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            info_dict['bboxes'].append(bbox)
    return info_dict

class_name_to_id_mapping = {
    "trafficlight": 0,
    "stop": 1,
    "speedlimit": 2,
    "crosswalk": 3
}

def convert_to_yolov5(info_dict, dataset_path):
    if info_dict.get('image_size') is None:
        print(f"Skipping {info_dict.get('filename', 'Unknown file')} due to missing image size info.")
        return

    print_buffer = []
    image_w, image_h, _ = info_dict["image_size"]

    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print(f"Invalid Class '{b['class']}' in {info_dict['filename']}. Must be one from {list(class_name_to_id_mapping.keys())}")
            continue

        b_center_x = (b["xmin"] + b["xmax"]) / 2
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width = b["xmax"] - b["xmin"]
        b_height = b["ymax"] - b["ymin"]

        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        print_buffer.append(f"{class_id} {b_center_x:.6f} {b_center_y:.6f} {b_width:.6f} {b_height:.6f}")

    filename_stem = os.path.splitext(info_dict["filename"])[0]
    save_file_name = os.path.join(dataset_path, "annotations", f"{filename_stem}.txt")

    with open(save_file_name, "w") as f:
        f.write("\n".join(print_buffer))

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), 'Road_Sign_Dataset')

    annotations_dir = os.path.join(dataset_path, 'annotations')
    if not os.path.isdir(annotations_dir):
        print(f"Annotations directory not found at: {annotations_dir}")
        sys.exit(1)

    annotations = [os.path.join(annotations_dir, x) for x in os.listdir(annotations_dir) if x.endswith(".xml")]
    annotations.sort()

    for ann in tqdm(annotations):
        info_dict = extract_info_from_xml(ann)
        convert_to_yolov5(info_dict, dataset_path)
    
    print("Annotation conversion to YOLOv5 format is complete.")
