# coding:utf-8
import argparse

from ultralytics import YOLO


def parse_args():
    """Parse training options and allow toggling the custom MSCA block."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 helmet detector.')
    parser.add_argument('--scale', default='n', choices=['n', 's', 'm', 'l', 'x'], help='YOLOv8 model scale.')
    parser.add_argument('--use-msca', action='store_true', help='Enable the custom MSCA attention block.')
    parser.add_argument('--weights', default=None, help='Pretrained weights path. Defaults to yolov8{scale}.pt.')
    parser.add_argument('--data', default='datasets/helmetData/data.yaml', help='Dataset YAML path.')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs.')
    parser.add_argument('--batch', type=int, default=8, help='Training batch size.')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size.')
    parser.add_argument('--device', default='0', help='Training device, such as 0 or cpu.')
    parser.add_argument('--cache', action='store_true', help='Cache images for faster training.')
    parser.add_argument('--workers', type=int, default=0, help='Dataloader worker count.')
    return parser.parse_args()


def get_model_cfg(scale, use_msca):
    """Return the base or MSCA-enhanced model YAML based on the selected switch."""
    suffix = '-msca' if use_msca else ''
    return f'ultralytics/cfg/models/v8/yolov8{scale}{suffix}.yaml'


if __name__ == '__main__':
    args = parse_args()
    model_cfg = get_model_cfg(args.scale, args.use_msca)
    weights = args.weights or f'yolov8{args.scale}.pt'

    model = YOLO(model_cfg).load(weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        cache=args.cache,
        workers=args.workers)

    # success = model.export(format='onnx')
