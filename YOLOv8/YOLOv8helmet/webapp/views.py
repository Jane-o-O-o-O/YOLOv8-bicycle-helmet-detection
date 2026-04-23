from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from PIL import Image, ImageFont
from ultralytics import YOLO

import Config
import detect_tools as tools

SUPPORTED_IMAGE_SUFFIX = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SUPPORTED_VIDEO_SUFFIX = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}


def _warmup_model(model: YOLO) -> None:
    try:
        model(np.zeros((48, 48, 3), dtype=np.uint8))
    except Exception:
        pass


def load_model() -> YOLO:
    try:
        import torch
        from ultralytics.nn.tasks import DetectionModel

        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([DetectionModel])
    except Exception:
        pass

    model = YOLO(Config.model_path, task='detect')
    _warmup_model(model)
    return model


def get_model():
    model = getattr(get_model, '_cached', None)
    if model is None:
        model = load_model()
        setattr(get_model, '_cached', model)
    return model


def load_rider_model() -> YOLO:
    model = YOLO(Config.rider_model_path, task='detect')
    _warmup_model(model)
    return model


def get_rider_model():
    model = getattr(get_rider_model, '_cached', None)
    if model is None:
        model = load_rider_model()
        setattr(get_rider_model, '_cached', model)
    return model


def _box_intersection_ratio(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter_area / area_a


def _detect_rider_violations(locations: list[list[int]], classes: list[int], rider_results) -> tuple[list[int], dict[int, int]]:
    two_wheeler_indices = [
        index for index, cls_id in enumerate(classes)
        if cls_id == Config.two_wheeler_class_id
    ]
    rider_counts = {index: 0 for index in two_wheeler_indices}
    if not two_wheeler_indices or rider_results is None:
        return [], rider_counts

    person_boxes: list[list[int]] = []
    for box, cls_id in zip(rider_results.boxes.xyxy.tolist(), rider_results.boxes.cls.tolist()):
        if int(cls_id) != Config.person_class_id:
            continue
        person_boxes.append([int(value) for value in box])

    for person_box in person_boxes:
        person_center = ((person_box[0] + person_box[2]) // 2, (person_box[1] + person_box[3]) // 2)
        for index in two_wheeler_indices:
            vehicle_box = locations[index]
            overlap_ratio = _box_intersection_ratio(person_box, vehicle_box)
            if overlap_ratio >= Config.rider_overlap_threshold or tools.point_in_rect(person_center, vehicle_box):
                rider_counts[index] += 1

    violation_indices = [
        index for index, count in rider_counts.items()
        if count >= Config.rider_min_person_count
    ]
    return violation_indices, rider_counts


def extract_detection_data(frame: np.ndarray, results, rider_results=None) -> dict:
    locations, classes, confidences = tools.filter_results(results)
    rider_violation_indices, rider_counts = _detect_rider_violations(locations, classes, rider_results)
    two_wheeler_count = sum(1 for cls_id in classes if cls_id == Config.two_wheeler_class_id)
    return {
        'locations': locations,
        'classes': classes,
        'confidences': confidences,
        'conf_texts': [f'{conf * 100:.2f}%' for conf in confidences],
        'violation_present': any(cls_id == Config.violation_class_id for cls_id in classes),
        'two_wheeler_count': two_wheeler_count,
        'rider_counts': rider_counts,
        'rider_violation_indices': rider_violation_indices,
        'rider_violation_present': bool(rider_violation_indices),
    }


def build_annotated_frame(frame, data):
    labels = []
    for index, (cls_id, conf) in enumerate(zip(data['classes'], data['confidences'])):
        label = f'{Config.CH_names[cls_id]} {conf * 100:.1f}%'
        if index in data['rider_violation_indices']:
            rider_count = data['rider_counts'].get(index, 0)
            label = f'{label} 载人:{rider_count}'
        labels.append(label)

    has_violation = data['violation_present'] or data['rider_violation_present']
    status_lines = [
        f'Targets: {len(data["locations"])}',
        f'Two-wheelers: {data["two_wheeler_count"]}',
        f'Helmet violation: {"YES" if data["violation_present"] else "NO"}',
        f'Passenger violation: {"YES" if data["rider_violation_present"] else "NO"}',
    ]

    font = ImageFont.truetype('Font/platech.ttf', 25, 0)
    colors = tools.Colors()
    return tools.draw_detection_frame(
        frame.copy(),
        data['locations'],
        data['classes'],
        labels,
        font,
        colors,
        status_lines=status_lines,
        stable_violation=has_violation,
        rider_violation_indices=data['rider_violation_indices'],
    )


def save_preview_image(image: np.ndarray, source_name: str) -> str:
    preview_dir = Path(settings.MEDIA_ROOT) / 'preview'
    preview_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = Path(str(source_name)).stem.replace(' ', '_')
    preview_path = preview_dir / f'{safe_stem}_preview.jpg'
    tools.img_cvwrite(str(preview_path), image)
    return f'{settings.MEDIA_URL}preview/{preview_path.name}'


def save_video_result_path(source_name: str) -> tuple[str, str]:
    video_dir = Path(settings.MEDIA_ROOT) / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = Path(str(source_name)).stem.replace(' ', '_')
    output_name = f'{safe_stem}_detect_result.mp4'
    output_path = video_dir / output_name
    output_url = f'{settings.MEDIA_URL}videos/{output_name}'
    return str(output_path), output_url


def save_evidence(image: np.ndarray, source_name: str) -> str:
    return tools.save_violation_event_image(image, source_name, Config.evidence_dir)


def append_violation_record(source_name: str, violation_type: str, confidence: str, location, evidence_path: str) -> None:
    if location is None:
        return
    tools.append_violation_record(
        Config.violation_record_path,
        source_name,
        violation_type,
        confidence,
        location,
        '-',
        evidence_path,
    )


def record_frame_violations(source_name: str, data: dict, evidence_path: str) -> int:
    record_count = 0
    helmet_violation_items = [
        (conf, location)
        for location, cls_id, conf in zip(data['locations'], data['classes'], data['confidences'])
        if cls_id == Config.violation_class_id
    ]
    if helmet_violation_items:
        best_conf, best_location = max(helmet_violation_items, key=lambda item: item[0])
        append_violation_record(
            source_name,
            Config.CH_names[Config.violation_class_id],
            f'{best_conf * 100:.2f}%',
            best_location,
            evidence_path,
        )
        record_count += 1

    for index in data['rider_violation_indices']:
        rider_count = data['rider_counts'].get(index, 0)
        append_violation_record(
            source_name,
            Config.passenger_violation_name,
            f'{rider_count}人',
            data['locations'][index],
            evidence_path,
        )
        record_count += 1

    return record_count


def load_records() -> pd.DataFrame:
    path = Path(Config.violation_record_path)
    if not path.exists():
        return pd.DataFrame(columns=['id', 'time', 'source', 'type', 'confidence', 'location', 'vote', 'evidence_path'])
    return pd.read_csv(path)


def dashboard(request: HttpRequest) -> HttpResponse:
    tools.ensure_dir(Config.save_path)
    tools.ensure_dir(Config.evidence_dir)
    model = None
    rider_model = None
    model_error = None
    try:
        model = get_model()
        rider_model = get_rider_model()
    except Exception as exc:
        model_error = str(exc)

    context = {
        'config': Config,
        'records': load_records().to_dict('records'),
        'class_options': [
            {'id': key, 'label': f'{key} - {Config.CH_names[key]}'}
            for key in Config.names.keys()
        ],
        'model_error': model_error,
        'last_image': None,
        'last_video': None,
        'folder_results': [],
        'preview_path': None,
    }

    if request.method == 'POST' and model is not None:
        action = request.POST.get('action')

        if action == 'image':
            upload = request.FILES.get('image_file')
            if upload:
                file_bytes = upload.read()
                img = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    messages.error(request, '无法读取上传的图片。')
                else:
                    result = model(img)[0]
                    rider_result = rider_model(img)[0] if rider_model is not None else None
                    data = extract_detection_data(img, result, rider_result)
                    annotated = build_annotated_frame(img, data)
                    preview_url = save_preview_image(annotated, upload.name)
                    evidence_path = '-'
                    has_violation = data['violation_present'] or data['rider_violation_present']
                    if has_violation:
                        evidence_path = save_evidence(annotated, upload.name)
                        record_frame_violations(upload.name, data, evidence_path)
                    context['last_image'] = {
                        'source': upload.name,
                        'targets': len(data['locations']),
                        'two_wheeler_count': data['two_wheeler_count'],
                        'helmet_violation': data['violation_present'],
                        'rider_violation': data['rider_violation_present'],
                        'rider_violation_count': len(data['rider_violation_indices']),
                        'violation': has_violation,
                        'preview_url': preview_url,
                        'evidence_path': evidence_path,
                    }
                    messages.success(request, f'图片检测完成，证据已保存：{evidence_path}')

        elif action == 'folder':
            folder_path = request.POST.get('folder_path', '').strip()
            if folder_path:
                path = Path(folder_path)
                if not path.exists() or not path.is_dir():
                    messages.error(request, '请输入正确的目录路径。')
                else:
                    results = []
                    for file in sorted(path.iterdir()):
                        if file.suffix.lower() not in SUPPORTED_IMAGE_SUFFIX:
                            continue
                        img = tools.img_cvread(str(file))
                        if img is None:
                            continue
                        result = model(img)[0]
                        rider_result = rider_model(img)[0] if rider_model is not None else None
                        data = extract_detection_data(img, result, rider_result)
                        annotated = build_annotated_frame(img, data)
                        out_name = f'{file.stem}_detect_result.jpg'
                        out_path = os.path.join(Config.save_path, out_name)
                        tools.img_cvwrite(out_path, annotated)
                        results.append({
                            'file': file.name,
                            'targets': len(data['locations']),
                            'two_wheeler_count': data['two_wheeler_count'],
                            'helmet_violation': data['violation_present'],
                            'rider_violation': data['rider_violation_present'],
                            'violation': data['violation_present'] or data['rider_violation_present'],
                            'output_path': out_path,
                        })
                    context['folder_results'] = results
                    messages.success(request, f'批量检测完成，共处理 {len(results)} 张图片。')

        elif action == 'video':
            video_file = request.FILES.get('video_file')
            if video_file:
                suffix = Path(video_file.name).suffix.lower()
                if suffix not in SUPPORTED_VIDEO_SUFFIX:
                    messages.error(request, '视频格式不支持。')
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        for chunk in video_file.chunks():
                            tmp.write(chunk)
                        temp_path = tmp.name
                    cap = cv2.VideoCapture(temp_path)
                    if not cap.isOpened():
                        messages.error(request, '无法读取上传的视频文件。')
                    else:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        if width <= 0 or height <= 0:
                            cap.release()
                            messages.error(request, '视频分辨率读取失败，请尝试更换视频格式。')
                        else:
                            out_path, out_url = save_video_result_path(video_file.name)
                            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                            frame_count = 0
                            saved_any = False
                            if not writer.isOpened():
                                messages.error(request, '视频写入器初始化失败，无法保存检测结果。')
                            else:
                                while cap.isOpened():
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    frame_count += 1
                                    result = model(frame)[0]
                                    rider_result = rider_model(frame)[0] if rider_model is not None else None
                                    data = extract_detection_data(frame, result, rider_result)
                                    annotated = build_annotated_frame(frame, data)
                                    writer.write(annotated)
                                    if (data['violation_present'] or data['rider_violation_present']) and not saved_any:
                                        evidence_path = save_evidence(annotated, video_file.name)
                                        record_frame_violations(video_file.name, data, evidence_path)
                                        saved_any = True
                                context['last_video'] = {'output_path': out_path, 'output_url': out_url, 'frames': frame_count}
                                messages.success(request, f'视频检测完成，结果保存为：{out_path}')
                            writer.release()
                            cap.release()
                    Path(temp_path).unlink(missing_ok=True)

        elif action == 'config':
            try:
                Config.violation_class_id = int(request.POST.get('violation_class_id', Config.violation_class_id))
                Config.two_wheeler_class_id = int(request.POST.get('two_wheeler_class_id', Config.two_wheeler_class_id))
                Config.person_class_id = int(request.POST.get('person_class_id', Config.person_class_id))
                Config.rider_min_person_count = int(request.POST.get('rider_min_person_count', Config.rider_min_person_count))
                Config.rider_overlap_threshold = float(request.POST.get('rider_overlap_threshold', Config.rider_overlap_threshold))
                Config.rider_model_path = request.POST.get('rider_model_path', Config.rider_model_path).strip() or Config.rider_model_path
                Config.save_path = request.POST.get('save_path', Config.save_path)
                Config.evidence_dir = os.path.join(Config.save_path, Config.evidence_dir_name)
                Config.violation_record_path = os.path.join(Config.save_path, Config.violation_record_name)
                setattr(get_rider_model, '_cached', None)
                tools.ensure_dir(Config.save_path)
                tools.ensure_dir(Config.evidence_dir)
                messages.success(request, '系统参数已更新。')
            except Exception as exc:
                messages.error(request, f'配置更新失败：{exc}')

    context['records'] = load_records().to_dict('records')
    return render(request, 'dashboard.html', context)
