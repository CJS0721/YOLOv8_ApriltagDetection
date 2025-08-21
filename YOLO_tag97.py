import os
import cv2
import numpy as np
from ultralytics import YOLO


def detect_tags():
    """使用YOLOv8模型检测图片，并将结果每9张合并输出（自动统一尺寸）"""

    # ====== 参数设置 ======
    input_folder = "extra_data"  # 输入图片文件夹
    output_folder = "ID97detect_results"  # 输出文件夹
    batch_output_folder = os.path.join(output_folder, "batch_results")  # 合并图片输出目录
    model_path = "runs/detect/train/weights/best.pt"  # 模型路径
    confidence_threshold = 0.5  # 置信度阈值
    batch_size = 9  # 每批次合并的图片数量
    target_size = (640, 640)  # 统一调整的目标尺寸 (宽度, 高度)
    # =====================

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(batch_output_folder, exist_ok=True)

    # 加载模型
    model = YOLO(model_path)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if not image_files:
        print(f"错误：{input_folder} 中没有找到图片文件！")
        return

    print(f"开始处理 {len(image_files)} 张图片...")

    # 分批处理图片
    for batch_idx in range(0, len(image_files), batch_size):
        batch_files = image_files[batch_idx:batch_idx + batch_size]
        batch_images = []

        # 批量检测并统一尺寸
        for img_file in batch_files:
            img_path = os.path.join(input_folder, img_file)

            # 检测并获取带标注框的图片
            results = model.predict(
                source=img_path,
                conf=confidence_threshold,
                save=False
            )
            plotted_img = results[0].plot()  # 获取检测后的NumPy数组 (BGR格式)

            # 统一调整图片尺寸
            resized_img = cv2.resize(plotted_img, target_size, interpolation=cv2.INTER_AREA)
            batch_images.append(resized_img)

            # 打印检测信息
            print(f"\n检测结果: {img_file}")
            for box in results[0].boxes:
                print(f"  → 标签位置: {box.xyxy[0].tolist()} | 置信度: {box.conf.item():.2f}")

        # 合并当前批次的图片
        if batch_images:
            # 计算合并后的网格大小（3x3）
            grid_size = int(np.ceil(np.sqrt(len(batch_images))))
            img_h, img_w = target_size[1], target_size[0]

            # 创建空白画布（白色背景）
            canvas = np.full((img_h * grid_size, img_w * grid_size, 3), 255, dtype=np.uint8)

            # 将图片填充到画布
            for i, img in enumerate(batch_images):
                row = i // grid_size
                col = i % grid_size
                canvas[row * img_h:(row + 1) * img_h, col * img_w:(col + 1) * img_w] = img

            # 保存合并后的图片
            output_path = os.path.join(batch_output_folder, f"batch_{batch_idx // batch_size + 1}.jpg")
            cv2.imwrite(output_path, canvas)

    print(f"\n处理完成！合并结果保存在: {batch_output_folder}")


if __name__ == "__main__":
    detect_tags()
