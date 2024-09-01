import xml.etree.ElementTree as ET
import os

# Define class mappings
class_mapping = {
    'entity': 0,
    'rel': 1,
    'rel_attr': 2,
    'many': 3,
    'one': 4,
    'weak_entity':5,
    'ident_rel':6
}

def convert_voc_to_yolo(xml_file, output_txt):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    with open(output_txt, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = class_mapping[class_name]

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Loop through and convert XML files
for i in range(1, 141):  # Loop from 1 to 140 inclusive
    xml_filename = f'{i:03}.xml'
    xml_path = os.path.join('Collection_labels_by_a_student/Collection_1', xml_filename)

    # Determine the output path based on the file number
    if i <= 112:
        out_folder = 'dataset/labels/train'
    else:
        out_folder = 'dataset/labels/val'

    out_path = os.path.join(out_folder, f'{i:03}.txt')

    # Handle missing files
    try:
        convert_voc_to_yolo(xml_path, out_path)
        print(f"Converted: {xml_filename} to {out_path}")
    except FileNotFoundError:
        print(f"{xml_filename} doesn't exist")
    except Exception as e:
        print(f"Error processing {xml_filename}: {e}")