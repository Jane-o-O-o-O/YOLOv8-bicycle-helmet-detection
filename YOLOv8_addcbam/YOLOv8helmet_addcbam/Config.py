# coding:utf-8
import os

# Detection output path
save_path = 'save_data'

# Model path
model_path = 'models/best.pt'

# Class names
names = {0: 'helmet', 1: 'without', 2: 'two_wheeler'}
CH_names = ['头盔', '未佩戴', '非机动车']

# ROI settings, values are relative ratios: left, top, right, bottom
roi_enabled = True
roi_ratio = (0.12, 0.12, 0.88, 0.95)

# Violation decision settings
violation_class_id = 1
vote_window = 10
vote_threshold = 6

# Evidence output
evidence_dir_name = 'violation_evidence'
violation_record_name = 'violation_records.csv'
evidence_dir = os.path.join(save_path, evidence_dir_name)
violation_record_path = os.path.join(save_path, violation_record_name)
