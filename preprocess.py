import json
import os
import os
import pickle
from argparse import ArgumentParser

import termcolor

from dataset.parsers import Mimic3Parser
from dataset.utils import encode_code, generate_code_levels, split_patients, generate_code_code_adjacent, build_code_xy, \
    generate_neighbors, divide_middle, build_heart_failure_y, save_data, normalize_adj, save_sparse

parser = ArgumentParser("Preprocess raw data")
parser.add_argument("--data_path", type=str, default="./data", help="Path to folder that contains icd9.txt")
parser.add_argument("--raw_path", type=str, default="../dataset/physionet.org/files/mimiciii/1.4", help="Path to mimiciii/1.4 folder")
args = parser.parse_args()

conf = {
    'parser': Mimic3Parser,
    'train_num': 6000,
    'test_num': 1000,
    'threshold': 0.01
}

raw_path = args.raw_path
data_path = args.data_path  # path to folder that contains icd9.txt
parsed_path = os.path.join(data_path, "parsed")
if not os.path.exists(parsed_path):
    os.makedirs(parsed_path)
parser = conf['parser'](raw_path)
sample_num = conf.get('sample_num', None)

"""parsing raw data"""
patient_admission, admission_codes = parser.parse(sample_num)
print(termcolor.colored('saving parsed data ...', 'green'))
if not os.path.exists(parsed_path):
    os.makedirs(parsed_path)

pickle.dump(patient_admission, open(os.path.join(parsed_path, 'patient_admission.pkl'), 'wb'))
pickle.dump(admission_codes, open(os.path.join(parsed_path, 'admission_codes.pkl'), 'wb'))

patient_num = len(patient_admission)  # number of patients
max_admission_num = max(
    [len(admissions) for admissions in patient_admission.values()])  # maximum number of admissions for a patient
avg_admission_num = sum([len(admissions) for admissions in
                         patient_admission.values()]) / patient_num  # average number of admissions for a patient
max_visit_code_num = max(
    [len(codes) for codes in admission_codes.values()])  # maximum number of disease codes for an admission
avg_visit_code_num = sum([len(codes) for codes in admission_codes.values()]) / len(
    admission_codes)  # average number of disease codes for an admission

# print with color
print(termcolor.colored('patient num: %d' % patient_num, 'blue'))
print(termcolor.colored('max admission num: %d' % max_admission_num, 'blue'))
print(termcolor.colored('mean admission num: %.2f' % avg_admission_num, 'blue'))
print(termcolor.colored('max code num in an admission: %d' % max_visit_code_num, 'blue'))
print(termcolor.colored('mean code num in an admission: %.2f' % avg_visit_code_num, 'blue'))

admission_codes_encoded, code_map = encode_code(patient_admission, admission_codes)
code_num = len(code_map)
print(termcolor.colored('There are %d disease codes' % code_num, 'blue'))

code_levels = generate_code_levels(data_path, code_map)
pickle.dump({
    'code_levels': code_levels,
}, open(os.path.join(parsed_path, 'code_levels.pkl'), 'wb'))

train_pids, valid_pids, test_pids = split_patients(
    patient_admission=patient_admission,
    admission_codes=admission_codes,
    code_map=code_map,
    train_num=conf['train_num'],
    test_num=conf['test_num']
)

print(termcolor.colored('train num: %d' % len(train_pids), 'blue'))
print(termcolor.colored('valid num: %d' % len(valid_pids), 'blue'))
print(termcolor.colored('test num: %d' % len(test_pids), 'blue'))

code_adj, sim_matrix = generate_code_code_adjacent(pids=train_pids,
                                       patient_admission=patient_admission,
                                       admission_codes_encoded=admission_codes_encoded,
                                       code_num=code_num,
                                       threshold=conf['threshold'], return_sim = True)
common_args = [patient_admission, admission_codes_encoded, max_admission_num, code_num]
print(termcolor.colored('building train codes features and labels ...', 'blue'))
train_code_x, train_codes_y, train_visit_lens, train_notes = build_code_xy(train_pids, *common_args)
print(termcolor.colored('building valid codes features and labels ...', 'blue'))
valid_code_x, valid_codes_y, valid_visit_lens, valid_notes = build_code_xy(valid_pids, *common_args)
print(termcolor.colored('building test codes features and labels ...', 'blue'))
test_code_x, test_codes_y, test_visit_lens, test_notes = build_code_xy(test_pids, *common_args)

print(termcolor.colored('generating train neighbors ...', 'blue'))
train_neighbors = generate_neighbors(train_code_x, train_visit_lens, code_adj)
print(termcolor.colored('generating valid neighbors ...', 'blue'))
valid_neighbors = generate_neighbors(valid_code_x, valid_visit_lens, code_adj)
print(termcolor.colored('generating test neighbors ...', 'blue'))
test_neighbors = generate_neighbors(test_code_x, test_visit_lens, code_adj)

print(termcolor.colored('generating train middles ...', 'blue'))
train_divided = divide_middle(train_code_x, train_neighbors, train_visit_lens)
print(termcolor.colored('generating valid middles ...', 'blue'))
valid_divided = divide_middle(valid_code_x, valid_neighbors, valid_visit_lens)
print(termcolor.colored('generating test middles ...', 'blue'))
test_divided = divide_middle(test_code_x, test_neighbors, test_visit_lens)

print(termcolor.colored('building train heart failure labels ...', 'blue'))
train_hf_y = build_heart_failure_y('428', train_codes_y, code_map)
print(termcolor.colored('building valid heart failure labels ...', 'blue'))
valid_hf_y = build_heart_failure_y('428', valid_codes_y, code_map)
print(termcolor.colored('building test heart failure labels ...', 'blue'))
test_hf_y = build_heart_failure_y('428', test_codes_y, code_map)

encoded_path = os.path.join(data_path, 'encoded')
if not os.path.exists(encoded_path):
    os.makedirs(encoded_path)

print(termcolor.colored('saving encoded data ...', 'green'))
pickle.dump(patient_admission, open(os.path.join(encoded_path, 'patient_admission.pkl'), 'wb'))
pickle.dump(admission_codes_encoded, open(os.path.join(encoded_path, 'codes_encoded.pkl'), 'wb'))
pickle.dump(code_map, open(os.path.join(encoded_path, 'code_map.pkl'), 'wb'))
pickle.dump({
    'train_pids': train_pids,
    'valid_pids': valid_pids,
    'test_pids': test_pids
}, open(os.path.join(encoded_path, 'pids.pkl'), 'wb'))

print(termcolor.colored('saving standard data ...', 'green'))
standard_path = os.path.join(data_path, 'standard')
train_path = os.path.join(standard_path, 'train')
valid_path = os.path.join(standard_path, 'valid')
test_path = os.path.join(standard_path, 'test')
if not os.path.exists(standard_path):
    os.makedirs(standard_path)

if not os.path.exists(train_path):
    os.makedirs(train_path)
    os.makedirs(valid_path)
    os.makedirs(test_path)

print(termcolor.colored('saving training data', 'green'))
save_data(train_path, train_code_x, train_visit_lens, train_codes_y, train_hf_y, train_divided, train_neighbors, train_notes)
print(termcolor.colored('saving valid data', 'green'))
save_data(valid_path, valid_code_x, valid_visit_lens, valid_codes_y, valid_hf_y, valid_divided, valid_neighbors, valid_notes)
print(termcolor.colored('saving test data', 'green'))
save_data(test_path, test_code_x, test_visit_lens, test_codes_y, test_hf_y, test_divided, test_neighbors, test_notes)

code_adj = normalize_adj(code_adj)  # normalized by node degree
sim_matrix = normalize_adj(sim_matrix)
save_sparse(os.path.join(standard_path, 'code_adj'), code_adj)
save_sparse(os.path.join(standard_path, 'sim_adj'), sim_matrix)

