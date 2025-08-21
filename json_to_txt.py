import json
import os

# 定义标签映射
label_map = {
    "tag": 0,
}


def convert_labelme_to_yolo(json_path, output_dir):
    with open(json_path, 'r') as f:
        labelme_data = json.load(f)

    image_width = labelme_data['imageWidth']
    image_height = labelme_data['imageHeight']

    yolo_annotations = []
    for shape in labelme_data['shapes']:
        label = shape['label']
        if label not in label_map:
            continue  # 忽略未定义的标签

        class_id = label_map[label]

        points = shape['points']
        if shape['shape_type'] == 'rectangle':
            (x1, y1), (x2, y2) = points
        elif shape['shape_type'] == 'polygon':
            x1, y1 = min(point[0] for point in points), min(point[1] for point in points)
            x2, y2 = max(point[0] for point in points), max(point[1] for point in points)
        else:
            continue  # 其他类型不处理

        x_center = (x1 + x2) / 2.0 / image_width
        y_center = (y1 + y2) / 2.0 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_path))[0] + '.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))


def process_folder(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 处理输入文件夹中的每个 JSON 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(input_folder, filename)
            convert_labelme_to_yolo(json_path, output_folder)


# 示例使用
input_folder = "datasets/tagID97/labels/val"
output_folder = "datasets/tagID97/val_txt"

process_folder(input_folder, output_folder)

# 列出输出文件夹中的文件以确认
os.listdir(output_folder)