# encoding:utf-8
import csv
import os
import re
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PyQt5.QtGui import QImage, QPixmap


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def drawRectBox(image, rect, addText, fontC, color):
    cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)

    text_width = max(80, 18 * max(2, len(addText)))
    cv2.rectangle(
        image,
        (rect[0] - 1, max(0, rect[1] - 28)),
        (rect[0] + text_width, rect[1]),
        color,
        -1,
        cv2.LINE_AA,
    )

    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((rect[0] + 2, rect[1] - 27), addText, (255, 255, 255), font=fontC)
    return np.array(img)


def img_cvread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def img_cvwrite(path, img):
    ensure_dir(os.path.dirname(path))
    ext = os.path.splitext(path)[1] or '.jpg'
    ok, buffer = cv2.imencode(ext, img)
    if not ok:
        return False
    buffer.tofile(path)
    return True


def draw_boxes(img, boxes):
    for each in boxes:
        x1, y1, x2, y2 = each
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def cvimg_to_qpiximg(cvimg):
    height, width, depth = cvimg.shape
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
    qimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
    return QPixmap(qimg)


def draw_roi_box(image, rect, label='ROI'):
    if rect is None:
        return image
    x1, y1, x2, y2 = rect
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 215, 255), 2)
    cv2.putText(image, label, (x1 + 6, max(24, y1 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
    return image


def draw_status_lines(image, lines):
    if not lines:
        return image
    y = 28
    for line in lines:
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(image, (8, y - text_h - 8), (20 + text_w, y + 6), (32, 32, 32), -1)
        cv2.putText(image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 30
    return image


def draw_detection_frame(image, locations, clses, labels, fontC, colors, roi_rect=None, status_lines=None, stable_violation=False):
    draw_roi_box(image, roi_rect)

    for location, cls_id, label in zip(locations, clses, labels):
        color = colors(int(cls_id), True)
        image = drawRectBox(image, location, label, fontC, color)

    banner = 'STABLE VIOLATION' if stable_violation else 'MONITORING'
    banner_color = (0, 0, 255) if stable_violation else (0, 170, 0)
    cv2.putText(
        image,
        banner,
        (12, max(40, image.shape[0] - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        banner_color,
        2,
    )
    return draw_status_lines(image, status_lines)


def get_roi_rect(image_shape, roi_ratio):
    if not roi_ratio:
        return None

    img_h, img_w = image_shape[:2]
    left, top, right, bottom = roi_ratio
    x1 = max(0, min(img_w - 1, int(img_w * left)))
    y1 = max(0, min(img_h - 1, int(img_h * top)))
    x2 = max(x1 + 1, min(img_w, int(img_w * right)))
    y2 = max(y1 + 1, min(img_h, int(img_h * bottom)))
    return [x1, y1, x2, y2]


def point_in_rect(point, rect):
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def filter_results(results, roi_rect=None):
    locations = []
    classes = []
    confidences = []

    for box, cls_id, conf in zip(
        results.boxes.xyxy.tolist(),
        results.boxes.cls.tolist(),
        results.boxes.conf.tolist(),
    ):
        location = [int(value) for value in box]
        if roi_rect:
            center = ((location[0] + location[2]) // 2, (location[1] + location[3]) // 2)
            if not point_in_rect(center, roi_rect):
                continue
        locations.append(location)
        classes.append(int(cls_id))
        confidences.append(float(conf))

    return locations, classes, confidences


def sanitize_filename(value):
    value = str(value).strip()
    value = re.sub(r'[\\/:*?"<>|]+', '_', value)
    return value or 'source'


def save_violation_event_image(image, source_name, evidence_dir):
    ensure_dir(evidence_dir)
    stem = sanitize_filename(os.path.splitext(os.path.basename(str(source_name)))[0])
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(evidence_dir, f'{stem}_{timestamp}.jpg')
    index = 1
    while os.path.exists(save_path):
        save_path = os.path.join(evidence_dir, f'{stem}_{timestamp}_{index}.jpg')
        index += 1
    img_cvwrite(save_path, image)
    return save_path


def insert_rows(path, lines, header):
    ensure_dir(os.path.dirname(path))

    no_header = not os.path.exists(path)
    if no_header:
        start_num = 1
    else:
        with open(path, 'r', encoding='utf-8-sig') as f:
            start_num = len(f.readlines())

    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        csv_write = csv.writer(f)
        if no_header:
            csv_write.writerow(header)

        for each_list in lines:
            csv_write.writerow([start_num] + each_list)
            start_num += 1


def append_violation_record(csv_path, source_name, cls_name, confidence, location, vote_text, evidence_path):
    header = ['id', 'time', 'source', 'type', 'confidence', 'location', 'vote', 'evidence_path']
    row = [[
        time.strftime('%Y-%m-%d %H:%M:%S'),
        str(source_name),
        str(cls_name),
        str(confidence),
        str(location),
        str(vote_text),
        str(evidence_path),
    ]]
    insert_rows(csv_path, row, header)


class Colors:
    def __init__(self):
        hexs = (
            'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231',
            '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', '00C2FF', '344593', '6473FF', '0018EC',
            '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7',
        )
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def yolo_to_location(w, h, yolo_data):
    x_, y_, w_, h_ = yolo_data
    x1 = int(w * x_ - 0.5 * w * w_)
    x2 = int(w * x_ + 0.5 * w * w_)
    y1 = int(h * y_ - 0.5 * h * h_)
    y2 = int(h * y_ + 0.5 * h * h_)
    return [x1, y1, x2, y2]


def location_to_yolo(w, h, locations):
    x1, y1, x2, y2 = locations
    x_ = float('%.5f' % ((x1 + x2) / 2 / w))
    y_ = float('%.5f' % ((y1 + y2) / 2 / h))
    w_ = float('%.5f' % ((x2 - x1) / w))
    h_ = float('%.5f' % ((y2 - y1) / h))
    return [x_, y_, w_, h_]


def draw_yolo_data(img_path, yolo_file_path):
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    with open(yolo_file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for each in data:
            temp = each.split()
            x_, y_, w_, h_ = eval(temp[1]), eval(temp[2]), eval(temp[3]), eval(temp[4])
            x1, y1, x2, y2 = yolo_to_location(w, h, [x_, y_, w_, h_])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))

    cv2.imshow('windows', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_path = 'TestFiles/1.jpg'
    yolo_file_path = 'save_data/yolo_labels/1.txt'
    draw_yolo_data(img_path, yolo_file_path)
