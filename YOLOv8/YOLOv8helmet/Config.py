# coding:utf-8
import os

# Detection output path
save_path = 'save_data'

# Model path
model_path = 'models/best.pt'
rider_model_path = 'yolov8n.pt'

# Class names
names = {0: 'helmet', 1: 'without', 2: 'two_wheeler'}
CH_names = ['头盔', '未佩戴', '非机动车']

# Compatibility flags for legacy desktop code paths.
# ROI is now disabled project-wide.
roi_enabled = False
roi_ratio = None

# Violation decision settings
violation_class_id = 1
two_wheeler_class_id = 2
person_class_id = 0
rider_min_person_count = 2
rider_overlap_threshold = 0.2
passenger_violation_name = '电动车载人'
vote_window = 10
vote_threshold = 6

# Evidence output
evidence_dir_name = 'violation_evidence'
violation_record_name = 'violation_records.csv'
evidence_dir = os.path.join(save_path, evidence_dir_name)
violation_record_path = os.path.join(save_path, violation_record_name)
