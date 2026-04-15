# -*- coding: utf-8 -*-
"""YOLO 数据集增强脚本。

默认支持标准 YOLO 目录结构：

    dataset/
      images/train
      images/val
      labels/train
      labels/val

也兼容你把图片和标签分别放在 `images` / `labels` 下面的情况。

增强策略：
- 水平翻转
- 轻微旋转
- 随机亮度/对比度
- 随机模糊
- 随机噪声

输出会写入新的目录，避免覆盖原始数据。
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class YoloBox:
    cls_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_list(self) -> List[float]:
        return [self.x_center, self.y_center, self.width, self.height]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment a YOLO detection dataset.")
    parser.add_argument("--input", required=True, help="Input dataset root directory.")
    parser.add_argument("--output", required=True, help="Output dataset root directory.")
    parser.add_argument("--copies", type=int, default=2, help="How many augmented copies to create per image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to augment, e.g. train. If images/train exists it will be used.",
    )
    parser.add_argument("--keep-original", action="store_true", help="Copy original images and labels too.")
    return parser.parse_args()


def find_split_dirs(root: Path, split: str) -> Tuple[Path, Path]:
    candidates = [
        (root / "images" / split, root / "labels" / split),
        (root / split / "images", root / split / "labels"),
        (root / "images", root / "labels"),
    ]
    for img_dir, label_dir in candidates:
        if img_dir.exists() and label_dir.exists():
            return img_dir, label_dir
    raise FileNotFoundError(
        f"Could not find image/label directories under {root}. Expected images/{split} and labels/{split}."
    )


def load_boxes(label_path: Path) -> List[YoloBox]:
    if not label_path.exists():
        return []
    boxes: List[YoloBox] = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        boxes.append(YoloBox(cls_id, x, y, w, h))
    return boxes


def save_boxes(label_path: Path, boxes: Sequence[YoloBox]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{b.cls_id} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}\n" for b in boxes]
    label_path.write_text("".join(lines), encoding="utf-8")


def yolo_to_xyxy(box: YoloBox, width: int, height: int) -> Tuple[float, float, float, float]:
    x1 = (box.x_center - box.width / 2) * width
    y1 = (box.y_center - box.height / 2) * height
    x2 = (box.x_center + box.width / 2) * width
    y2 = (box.y_center + box.height / 2) * height
    return x1, y1, x2, y2


def xyxy_to_yolo(cls_id: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Optional[YoloBox]:
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    return YoloBox(
        cls_id=cls_id,
        x_center=((x1 + x2) / 2) / width,
        y_center=((y1 + y2) / 2) / height,
        width=bw / width,
        height=bh / height,
    )


def flip_boxes_horizontal(boxes: Sequence[YoloBox]) -> List[YoloBox]:
    return [YoloBox(b.cls_id, 1.0 - b.x_center, b.y_center, b.width, b.height) for b in boxes]


def rotate_boxes(
    boxes: Sequence[YoloBox],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> List[YoloBox]:
    rotated: List[YoloBox] = []
    for box in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(box, width, height)
        corners = np.array(
            [[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32
        )
        transformed = (matrix @ corners.T).T
        xs = transformed[:, 0]
        ys = transformed[:, 1]
        new_box = xyxy_to_yolo(box.cls_id, float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()), width, height)
        if new_box is not None:
            rotated.append(new_box)
    return rotated


def transform_image_and_boxes(
    image: np.ndarray,
    boxes: Sequence[YoloBox],
    rng: random.Random,
) -> Tuple[np.ndarray, List[YoloBox]]:
    h, w = image.shape[:2]
    result = image.copy()
    result_boxes = list(boxes)

    if rng.random() < 0.5:
        result = cv2.flip(result, 1)
        result_boxes = flip_boxes_horizontal(result_boxes)

    angle = rng.uniform(-10.0, 10.0)
    scale = rng.uniform(0.95, 1.05)
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    result = cv2.warpAffine(result, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    result_boxes = rotate_boxes(result_boxes, matrix, w, h)

    alpha = rng.uniform(0.85, 1.15)  # contrast
    beta = rng.uniform(-20, 20)  # brightness
    result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        result = cv2.GaussianBlur(result, (k, k), 0)

    if rng.random() < 0.3:
        noise = rng.normalvariate(0, 8)
        gaussian = rng.normal(0, max(1.0, abs(noise)), result.shape).astype(np.float32)
        noisy = result.astype(np.float32) + gaussian
        result = np.clip(noisy, 0, 255).astype(np.uint8)

    return result, result_boxes


def iter_images(img_dir: Path) -> Iterable[Path]:
    for path in sorted(img_dir.rglob("*")):
        if path.suffix.lower() in IMG_EXTENSIONS:
            yield path


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    input_root = Path(args.input)
    output_root = Path(args.output)
    img_dir, label_dir = find_split_dirs(input_root, args.split)

    out_img_dir = output_root / "images" / args.split
    out_label_dir = output_root / "labels" / args.split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_images(img_dir))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        boxes = load_boxes(label_path)

        if args.keep_original:
            cv2.imwrite(str(out_img_dir / image_path.name), image)
            save_boxes(out_label_dir / f"{image_path.stem}.txt", boxes)

        for idx in range(args.copies):
            aug_image, aug_boxes = transform_image_and_boxes(image, boxes, rng)
            out_name = f"{image_path.stem}_aug{idx+1}{image_path.suffix}"
            cv2.imwrite(str(out_img_dir / out_name), aug_image)
            save_boxes(out_label_dir / f"{image_path.stem}_aug{idx+1}.txt", aug_boxes)

    print(f"Done. Augmented dataset saved to: {output_root}")


if __name__ == "__main__":
    main()
