# -*- coding: utf-8 -*-
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from PIL import ImageFont
from PyQt5.QtCore import QCoreApplication, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QLabel,
    QHeaderView,
    QMainWindow,
    QMessageBox,
    QTableWidgetItem,
)
from ultralytics import YOLO

import Config
import detect_tools as tools

sys.path.append('UIProgram')
from UIProgram.QssLoader import QSSLoader
from UIProgram.UiMain import Ui_MainWindow
from UIProgram.precess_bar import ProgressBar


def extract_detection_data(frame, results):
    roi_rect = tools.get_roi_rect(frame.shape, Config.roi_ratio) if Config.roi_enabled else None
    locations, classes, confidences = tools.filter_results(results, roi_rect)
    return {
        'roi_rect': roi_rect,
        'locations': locations,
        'classes': classes,
        'confidences': confidences,
        'conf_texts': [f'{conf * 100:.2f} %' for conf in confidences],
        'violation_present': any(cls_id == Config.violation_class_id for cls_id in classes),
    }


def build_status_lines(target_count, vote_hits, stable_violation, violation_present, enable_vote):
    lines = [
        f'ROI: {"ON" if Config.roi_enabled else "OFF"}',
        f'Targets in ROI: {target_count}',
    ]
    if enable_vote:
        lines.append(f'Vote: {vote_hits}/{Config.vote_window}')
        lines.append(f'Stable violation: {"YES" if stable_violation else "NO"}')
    else:
        lines.append(f'Violation target: {"YES" if violation_present else "NO"}')
    return lines


def build_annotated_frame(frame, data, fontC, colors, vote_hits=0, stable_violation=False, enable_vote=False):
    labels = [
        f'{Config.CH_names[cls_id]} {conf * 100:.1f}%'
        for cls_id, conf in zip(data['classes'], data['confidences'])
    ]
    status_lines = build_status_lines(
        len(data['locations']),
        vote_hits,
        stable_violation,
        data['violation_present'],
        enable_vote,
    )
    annotated = tools.draw_detection_frame(
        frame.copy(),
        data['locations'],
        data['classes'],
        labels,
        fontC,
        colors,
        roi_rect=data['roi_rect'],
        status_lines=status_lines,
        stable_violation=stable_violation,
    )
    return annotated, status_lines


def record_violation_event(image, source_name, data, vote_hits):
    violation_items = [
        (conf, location)
        for location, cls_id, conf in zip(data['locations'], data['classes'], data['confidences'])
        if cls_id == Config.violation_class_id
    ]
    if not violation_items:
        return ''

    best_conf, best_location = max(violation_items, key=lambda item: item[0])
    evidence_path = tools.save_violation_event_image(image, source_name, Config.evidence_dir)
    tools.append_violation_record(
        Config.violation_record_path,
        source_name,
        Config.CH_names[Config.violation_class_id],
        f'{best_conf * 100:.2f} %',
        best_location,
        f'{vote_hits}/{Config.vote_window}',
        evidence_path,
    )
    return evidence_path


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initMain()
        self.signalconnect()

        style_file = 'UIProgram/style.css'
        qssStyleSheet = QSSLoader.read_qss_file(style_file)
        self.setStyleSheet(qssStyleSheet)

    def signalconnect(self):
        self.ui.PicBtn.clicked.connect(self.open_img)
        self.ui.comboBox.activated.connect(self.combox_change)
        self.ui.VideoBtn.clicked.connect(self.vedio_show)
        self.ui.SaveBtn.clicked.connect(self.save_detect_video)
        self.ui.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.ui.FilesBtn.clicked.connect(self.detact_batch_imgs)

    def initMain(self):
        self.show_width = 770
        self.show_height = 480
        self.org_path = None
        self.org_img = None
        self.cap = None
        self.is_camera_open = False
        self.current_video_source = ''

        tools.ensure_dir(Config.save_path)
        tools.ensure_dir(Config.evidence_dir)

        self.model = YOLO(Config.model_path, task='detect')
        self.model(np.zeros((48, 48, 3)))
        self.fontC = ImageFont.truetype('Font/platech.ttf', 25, 0)
        self.colors = tools.Colors()

        self.timer_camera = QTimer(self)
        self.timer_camera.timeout.connect(self.open_frame)

        # Real-time camera monitoring is removed from this build.
        self.ui.CapBtn.hide()
        self.ui.CaplineEdit.hide()

        self.init_table()
        self.init_innovation_panel()
        self.reset_display_state()
        self.reset_vote_state()
        self.statusBar().showMessage('ROI已启用，等待检测。')

    def init_innovation_panel(self):
        self.violation_event_count = 0
        self.last_evidence_path = '暂无'

        # Compress the lower right area slightly and insert a visible innovation panel.
        self.ui.groupBox_2.setGeometry(0, 180, 431, 335)
        self.ui.frame_6.setGeometry(0, 155, 431, 170)

        self.innovation_group = self._create_group_box(self.ui.frame_4, 0, 520, 431, 90, '创新点展示')
        self.roi_status_lb = self._create_info_label(self.innovation_group, 16, 28, 190, 'ROI状态: 已启用')
        self.vote_status_lb = self._create_info_label(self.innovation_group, 220, 28, 190, f'投票结果: 0/{Config.vote_window}')
        self.stable_status_lb = self._create_info_label(self.innovation_group, 16, 52, 190, '稳定违规: 否')
        self.violation_count_lb = self._create_info_label(self.innovation_group, 220, 52, 190, '累计违规: 0')
        self.evidence_status_lb = self._create_info_label(self.innovation_group, 16, 72, 395, '最近证据: 暂无', point_size=10)

        self.ui.groupBox_4.setGeometry(0, 615, 431, 90)
        self.ui.SaveBtn.setGeometry(30, 35, 151, 40)
        self.ui.ExitBtn.setGeometry(250, 35, 151, 40)

    def _create_group_box(self, parent, x, y, w, h, title):
        box = self.ui.groupBox_4.__class__(parent)
        box.setGeometry(x, y, w, h)
        box.setFont(self.ui.groupBox_4.font())
        box.setTitle(title)
        return box

    def _create_info_label(self, parent, x, y, w, text, point_size=11):
        label = QLabel(parent)
        label.setGeometry(x, y, w, 20)
        font = label.font()
        font.setPointSize(point_size)
        label.setFont(font)
        label.setText(text)
        return label

    def update_innovation_panel(self, stable_violation=False, vote_enabled=False):
        roi_text = '已启用' if Config.roi_enabled else '已关闭'
        vote_text = f'{self.vote_hits}/{Config.vote_window}' if vote_enabled else '-'
        stable_text = '是' if stable_violation else '否'
        evidence_text = self.last_evidence_path if self.last_evidence_path else '暂无'
        if len(evidence_text) > 34:
            evidence_text = '...' + evidence_text[-31:]

        self.roi_status_lb.setText(f'ROI状态: {roi_text}')
        self.vote_status_lb.setText(f'投票结果: {vote_text}')
        self.stable_status_lb.setText(f'稳定违规: {stable_text}')
        self.violation_count_lb.setText(f'累计违规: {self.violation_event_count}')
        self.evidence_status_lb.setText(f'最近证据: {evidence_text}')

    def init_table(self):
        self.ui.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(40)
        self.ui.tableWidget.setColumnWidth(0, 80)
        self.ui.tableWidget.setColumnWidth(1, 200)
        self.ui.tableWidget.setColumnWidth(2, 150)
        self.ui.tableWidget.setColumnWidth(3, 90)
        self.ui.tableWidget.setColumnWidth(4, 230)
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ui.tableWidget.verticalHeader().setVisible(False)
        self.ui.tableWidget.setAlternatingRowColors(True)

    def reset_display_state(self):
        self.location_list = []
        self.cls_list = []
        self.conf_scores = []
        self.conf_list = []
        self.draw_img = None
        self.raw_view_image = None
        self.roi_rect = None
        self.status_lines = []
        self.current_source_name = ''

    def reset_vote_state(self):
        self.vote_history = deque(maxlen=Config.vote_window)
        self.vote_hits = 0
        self.event_active = False
        self.update_innovation_panel(stable_violation=False, vote_enabled=False)

    def stop_current_capture(self):
        if self.cap is not None:
            self.video_stop()
        self.is_camera_open = False
        self.current_video_source = ''

    def run_model(self, image_or_path):
        t1 = time.time()
        results = self.model(image_or_path)[0]
        self.ui.time_lb.setText('{:.3f} s'.format(time.time() - t1))
        return results

    def show_image(self, image):
        self.draw_img = image
        self.img_width, self.img_height = self.get_resize_size(image)
        resize_cvimg = cv2.resize(image, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

    def update_combo_items(self):
        choose_list = ['全部']
        choose_list.extend(
            f'{Config.names[cls_id]}_{index}'
            for index, cls_id in enumerate(self.cls_list)
        )
        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(choose_list)

    def update_detail_panel(self):
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))
        if target_nums >= 1:
            self.ui.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.ui.label_conf.setText(self.conf_list[0])
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.ui.type_lb.setText('')
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

    def display_detection(self, frame, source_name, data, append_table=False, stable_violation=False, enable_vote=False):
        annotated, self.status_lines = build_annotated_frame(
            frame,
            data,
            self.fontC,
            self.colors,
            vote_hits=self.vote_hits,
            stable_violation=stable_violation,
            enable_vote=enable_vote,
        )
        self.raw_view_image = frame.copy()
        self.roi_rect = data['roi_rect']
        self.location_list = data['locations']
        self.cls_list = data['classes']
        self.conf_scores = data['confidences']
        self.conf_list = data['conf_texts']
        self.current_source_name = str(source_name)

        if not append_table:
            self.ui.tableWidget.setRowCount(0)
            self.ui.tableWidget.clearContents()

        self.show_image(annotated)
        self.update_combo_items()
        self.update_detail_panel()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=source_name)
        self.update_innovation_panel(stable_violation=stable_violation, vote_enabled=enable_vote)
        self.statusBar().showMessage(' | '.join(self.status_lines))

    def handle_vote_and_record(self, source_name, data):
        self.vote_history.append(1 if data['violation_present'] else 0)
        self.vote_hits = sum(self.vote_history)
        stable_violation = (
            len(self.vote_history) >= Config.vote_threshold
            and self.vote_hits >= Config.vote_threshold
        )

        if stable_violation and not self.event_active:
            event_image = self.raw_view_image if self.raw_view_image is not None else self.org_img
            annotated, _ = build_annotated_frame(
                event_image,
                data,
                self.fontC,
                self.colors,
                vote_hits=self.vote_hits,
                stable_violation=True,
                enable_vote=True,
            )
            evidence_path = record_violation_event(annotated, source_name, data, self.vote_hits)
            self.event_active = True
            if evidence_path:
                self.violation_event_count += 1
                self.last_evidence_path = evidence_path
                self.statusBar().showMessage(f'违规证据已保存: {evidence_path}')
        elif not stable_violation:
            self.event_active = False

        return stable_violation

    def detect_and_display(self, frame, source_name, results, append_table=False, enable_vote=False):
        data = extract_detection_data(frame, results)
        stable_violation = data['violation_present']
        if enable_vote:
            stable_violation = self.handle_vote_and_record(source_name, data)
        else:
            self.vote_hits = 1 if data['violation_present'] else 0
            self.event_active = False

        self.display_detection(
            frame,
            source_name,
            data,
            append_table=append_table,
            stable_violation=stable_violation,
            enable_vote=enable_vote,
        )

    def build_static_result_image(self, img_path):
        image = tools.img_cvread(img_path)
        results = self.model(img_path)[0]
        data = extract_detection_data(image, results)
        annotated, _ = build_annotated_frame(
            image,
            data,
            self.fontC,
            self.colors,
            vote_hits=1 if data['violation_present'] else 0,
            stable_violation=data['violation_present'],
            enable_vote=False,
        )
        return annotated

    def open_img(self):
        self.stop_current_capture()
        file_path, _ = QFileDialog.getOpenFileName(None, '打开图片', './', 'Image files (*.jpg *.jpeg *.png)')
        if not file_path:
            return

        self.ui.comboBox.setDisabled(False)
        self.reset_vote_state()
        self.org_path = file_path
        self.org_img = tools.img_cvread(self.org_path)
        self.ui.PiclineEdit.setText(self.org_path)
        results = self.run_model(self.org_path)
        self.detect_and_display(self.org_img, self.org_path, results, append_table=False, enable_vote=False)

    def detact_batch_imgs(self):
        self.stop_current_capture()
        directory = QFileDialog.getExistingDirectory(self, '选择文件夹', './')
        if not directory:
            return

        self.ui.comboBox.setDisabled(False)
        self.reset_vote_state()
        self.org_path = directory
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()

        img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
        has_image = False
        for file_name in os.listdir(directory):
            full_path = os.path.join(directory, file_name)
            if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                has_image = True
                image = tools.img_cvread(full_path)
                results = self.run_model(full_path)
                self.detect_and_display(image, full_path, results, append_table=True, enable_vote=False)
                self.ui.PiclineEdit.setText(full_path)
                self.ui.tableWidget.scrollToBottom()
                QApplication.processEvents()

        if not has_image:
            QMessageBox.about(self, '提示', '所选文件夹中没有可检测图片。')

    def combox_change(self):
        if self.raw_view_image is None:
            return

        if not self.location_list:
            self.show_image(tools.draw_detection_frame(
                self.raw_view_image.copy(),
                [],
                [],
                [],
                self.fontC,
                self.colors,
                roi_rect=self.roi_rect,
                status_lines=self.status_lines,
                stable_violation=False,
            ))
            return

        com_text = self.ui.comboBox.currentText()
        if com_text == '全部':
            selected_locations = self.location_list
            selected_classes = self.cls_list
            selected_confidences = self.conf_scores
        else:
            selected_index = int(com_text.split('_')[-1])
            selected_locations = [self.location_list[selected_index]]
            selected_classes = [self.cls_list[selected_index]]
            selected_confidences = [self.conf_scores[selected_index]]

        labels = [
            f'{Config.CH_names[cls_id]} {conf * 100:.1f}%'
            for cls_id, conf in zip(selected_classes, selected_confidences)
        ]
        preview = tools.draw_detection_frame(
            self.raw_view_image.copy(),
            selected_locations,
            selected_classes,
            labels,
            self.fontC,
            self.colors,
            roi_rect=self.roi_rect,
            status_lines=self.status_lines,
            stable_violation=False,
        )
        self.show_image(preview)

        self.ui.type_lb.setText(Config.CH_names[selected_classes[0]])
        self.ui.label_conf.setText(f'{selected_confidences[0] * 100:.2f} %')
        self.ui.label_xmin.setText(str(selected_locations[0][0]))
        self.ui.label_ymin.setText(str(selected_locations[0][1]))
        self.ui.label_xmax.setText(str(selected_locations[0][2]))
        self.ui.label_ymax.setText(str(selected_locations[0][3]))

    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', 'Video files (*.avi *.mp4 *.mov)')
        if not file_path:
            return None
        self.org_path = file_path
        self.current_video_source = file_path
        self.ui.VideolineEdit.setText(file_path)
        return file_path

    def video_start(self):
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.ui.comboBox.clear()
        self.reset_vote_state()
        self.timer_camera.start(1)

    def tabel_info_show(self, locations, clses, confs, path=None):
        for location, cls, conf in zip(locations, clses, confs):
            row_count = self.ui.tableWidget.rowCount()
            self.ui.tableWidget.insertRow(row_count)

            item_id = QTableWidgetItem(str(row_count + 1))
            item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            item_path = QTableWidgetItem(str(path))

            item_cls = QTableWidgetItem(str(Config.CH_names[cls]))
            item_cls.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            item_conf = QTableWidgetItem(str(conf))
            item_conf.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

            item_location = QTableWidgetItem(str(location))

            self.ui.tableWidget.setItem(row_count, 0, item_id)
            self.ui.tableWidget.setItem(row_count, 1, item_path)
            self.ui.tableWidget.setItem(row_count, 2, item_cls)
            self.ui.tableWidget.setItem(row_count, 3, item_conf)
            self.ui.tableWidget.setItem(row_count, 4, item_location)
        self.ui.tableWidget.scrollToBottom()

    def video_stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.timer_camera.stop()

    def open_frame(self):
        if self.cap is None:
            return

        ret, now_img = self.cap.read()
        if not ret:
            self.video_stop()
            return

        self.raw_view_image = now_img.copy()
        results = self.run_model(now_img)
        self.detect_and_display(now_img, self.current_video_source, results, append_table=False, enable_vote=False)

    def vedio_show(self):
        video_path = self.get_video_path()
        if not video_path:
            return

        self.cap = cv2.VideoCapture(video_path)
        self.ui.comboBox.setDisabled(True)
        self.video_start()

    def get_resize_size(self, img):
        img_height, img_width, _ = img.shape
        ratio = img_width / img_height
        if ratio >= self.show_width / self.show_height:
            self.img_width = self.show_width
            self.img_height = int(self.img_width / ratio)
        else:
            self.img_height = self.show_height
            self.img_width = int(self.img_height * ratio)
        return self.img_width, self.img_height

    def save_detect_video(self):
        if self.cap is None and not self.org_path:
            QMessageBox.about(self, '提示', '当前没有可保存的信息，请先打开图片或视频。')
            return

        tools.ensure_dir(Config.save_path)

        if self.cap:
            res = QMessageBox.information(
                self,
                '提示',
                '保存视频检测结果可能需要较长时间，是否继续？',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if res != QMessageBox.Yes:
                return

            self.video_stop()
            self.btn2Thread_object = btn2Thread(self.org_path)
            self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
            self.btn2Thread_object.start()
            return

        if os.path.isfile(self.org_path):
            fileName = os.path.basename(self.org_path)
            name, end_name = fileName.rsplit('.', 1)
            save_img_path = os.path.join(Config.save_path, name + '_detect_result.' + end_name)
            if self.draw_img is not None and tools.img_cvwrite(save_img_path, self.draw_img):
                QMessageBox.about(self, '提示', f'图片保存成功!\n文件路径:{save_img_path}')
            return

        img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
        for file_name in os.listdir(self.org_path):
            full_path = os.path.join(self.org_path, file_name)
            if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                name, end_name = file_name.rsplit('.', 1)
                save_img_path = os.path.join(Config.save_path, name + '_detect_result.' + end_name)
                now_img = self.build_static_result_image(full_path)
                tools.img_cvwrite(save_img_path, now_img)

        QMessageBox.about(self, '提示', f'图片保存成功!\n文件路径:{Config.save_path}')

    def update_process_bar(self, cur_num, total):
        if cur_num == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()

        if cur_num >= total:
            self.progress_bar.close()
            QMessageBox.about(self, '提示', f'视频保存成功!\n文件位于 {Config.save_path}')
            return

        if self.progress_bar.isVisible() is False:
            self.btn2Thread_object.stop()
            return

        value = int(cur_num / total * 100)
        self.progress_bar.setValue(cur_num, total, value)
        QApplication.processEvents()


class btn2Thread(QThread):
    update_ui_signal = pyqtSignal(int, int)

    def __init__(self, path):
        super(btn2Thread, self).__init__()
        self.org_path = path
        self.is_running = True

    def run(self):
        cap = cv2.VideoCapture(self.org_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        fileName = os.path.basename(self.org_path)
        name = os.path.splitext(fileName)[0]
        save_video_path = os.path.join(Config.save_path, name + '_detect_result.avi')
        out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

        model = YOLO(Config.model_path, task='detect')
        fontC = ImageFont.truetype('Font/platech.ttf', 25, 0)
        colors = tools.Colors()
        vote_history = deque(maxlen=Config.vote_window)
        event_active = False

        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cur_num = 0

        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            cur_num += 1
            results = model(frame)[0]
            data = extract_detection_data(frame, results)
            vote_history.append(1 if data['violation_present'] else 0)
            vote_hits = sum(vote_history)
            stable_violation = (
                len(vote_history) >= Config.vote_threshold
                and vote_hits >= Config.vote_threshold
            )

            annotated, _ = build_annotated_frame(
                frame,
                data,
                fontC,
                colors,
                vote_hits=vote_hits,
                stable_violation=stable_violation,
                enable_vote=True,
            )

            if stable_violation and not event_active:
                record_violation_event(annotated, self.org_path, data, vote_hits)
                event_active = True
            elif not stable_violation:
                event_active = False

            out.write(annotated)
            self.update_ui_signal.emit(cur_num, total)

        cap.release()
        out.release()

    def stop(self):
        self.is_running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
