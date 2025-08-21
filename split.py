import os
import shutil
import random
from pathlib import Path

random.seed(0)


def split_data(img_dir, json_dir, output_dir, train_rate=0.7, val_rate=0.1, test_rate=0.2):
    """
    新版数据划分函数（支持JSON标注）
    输出目录结构:
    output_dir/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
    """
    # 验证比例总和
    assert abs(train_rate + val_rate + test_rate - 1.0) < 1e-6, "比例总和必须等于1"

    # 获取文件列表（兼容大小写）
    images = [f for f in os.listdir(img_dir) if Path(f).suffix.lower() in ['.png', '.jpg', '.jpeg']]
    labels = [f for f in os.listdir(json_dir) if Path(f).suffix.lower() == '.json']

    # 文件名匹配
    img_stems = {Path(f).stem: f for f in images}
    label_stems = {Path(f).stem: f for f in labels}
    common_stems = set(img_stems.keys()) & set(label_stems.keys())

    if not common_stems:
        raise ValueError("没有匹配的图片和标注文件！请检查文件名是否对应（如 0001.jpg 对应 0001.json）")

    # 配对数据并打乱
    paired_data = [(img_stems[name], label_stems[name]) for name in common_stems]
    random.shuffle(paired_data)
    total = len(paired_data)

    # 计算划分点
    train_end = int(train_rate * total)
    val_end = train_end + int(val_rate * total)

    # 划分数据集
    train_data = paired_data[:train_end]
    val_data = paired_data[train_end:val_end]
    test_data = paired_data[val_end:]

    # 创建主目录结构
    img_output_root = os.path.join(output_dir, 'images')
    label_output_root = os.path.join(output_dir, 'labels')

    # 划分集目录
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    # 先清空输出目录（如果存在）
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 复制文件到新结构
    for split_name, data in splits.items():
        # 创建图片子目录
        img_split_dir = os.path.join(img_output_root, split_name)
        os.makedirs(img_split_dir, exist_ok=True)

        # 创建标注子目录
        label_split_dir = os.path.join(label_output_root, split_name)
        os.makedirs(label_split_dir, exist_ok=True)

        # 复制文件
        for img_file, label_file in data:
            shutil.copy(
                os.path.join(img_dir, img_file),
                os.path.join(img_split_dir, img_file)
            )
            shutil.copy(
                os.path.join(json_dir, label_file),
                os.path.join(label_split_dir, label_file)
            )

    # 打印统计信息
    print("\n✅ 数据划分完成")
    print(f"总样本数: {total}")
    print(f"训练集: {len(train_data)} ({len(train_data) / total:.1%})")
    print(f"验证集: {len(val_data)} ({len(val_data) / total:.1%})")
    print(f"测试集: {len(test_data)} ({len(test_data) / total:.1%})")
    print(f"\n输出目录结构:\n{Path(output_dir).resolve()}")


if __name__ == '__main__':
    # 配置路径
    img_dir = "datasets/images"
    json_dir = "datasets/annotations"
    output_dir = "datasets/split_data"  # 新输出目录

    split_data(
        img_dir=img_dir,
        json_dir=json_dir,
        output_dir=output_dir,
        train_rate=0.7,
        val_rate=0.1,
        test_rate=0.2
    )
