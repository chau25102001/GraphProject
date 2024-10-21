import os
from collections import OrderedDict

import numpy as np
import termcolor
from tqdm import tqdm

from dataset.parsers import EHRParser
import torch

def encode_code(patient_admission: dict, admission_codes: dict):
    """
    This function encodes the disease codes in admission_codes into integers
    :param patient_admission: dict, key by patient ID, value by list of admissions
    :param admission_codes: dict, key by admission ID, value by list of disease codes
    :return:
    """
    code_map = OrderedDict()
    for pid, admissions in patient_admission.items():
        for admission in admissions:
            codes = admission_codes[admission[EHRParser.adm_id_col]]
            for code in codes:
                if code not in code_map:
                    code_map[code] = len(code_map)

    admission_codes_encoded = {
        admission_id: list(set(code_map[code] for code in codes))
        for admission_id, codes in admission_codes.items()
    }
    return admission_codes_encoded, code_map


def parse_icd9_range(range_: str) -> (str, str, int, int):
    """
    This function parses the string of ICD-9 range in icd9.txt file to (prefix, format, start, end)
    Examples:
    "001-009" -> ('', '%03d', 1, 9)
    "V01-V09" -> ('V', '%02d', 1, 9)
    "E800-E999" -> ('E', '%03d', 800, 999)
    :param range_:
    :return:
    """
    ranges = range_.lstrip().split('-')
    if ranges[0][0] == 'V':
        prefix = 'V'
        format_ = '%02d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    elif ranges[0][0] == 'E':
        prefix = 'E'
        format_ = '%03d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    else:
        prefix = ''
        format_ = '%03d'
        if len(ranges) == 1:
            start = int(ranges[0])
            end = start
        else:
            start, end = int(ranges[0]), int(ranges[1])
    return prefix, format_, start, end


def generate_code_levels(path, code_map: dict) -> np.ndarray:
    """
    This function generates the levels of disease codes based on the idc9 file and append with the encoded disease code integer
    :param path: str, path to the data folder
    :param code_map: dict, key by disease code, value by integer
    :return: np.ndarray
    """
    print(termcolor.colored('generating code levels ...', 'green'))
    three_level_code_set = set(code.split('.')[0] for code in code_map)
    icd9_path = os.path.join(path, 'icd9.txt')
    icd9_range = list(open(icd9_path, 'r', encoding='utf-8').readlines())
    three_level_dict = dict()
    level1, level2, level3 = (0, 0, 0)
    level1_can_add = False
    for range_ in icd9_range:
        range_ = range_.rstrip()
        if range_.startswith(" "):
            prefix, format_, start, end = parse_icd9_range(range_)
            level2_cannot_add = True
            for i in range(start, end + 1):
                code = prefix + format_ % i
                if code in three_level_code_set:
                    three_level_dict[code] = [level1, level2, level3]
                    level3 += 1
                    level1_can_add = True
                    level2_cannot_add = False
            if not level2_cannot_add:
                level2 += 1
        else:
            if level1_can_add:
                level1 += 1
                level1_can_add = False

    code_level = dict()
    for code, cid in code_map.items():
        three_level_code = code.split('.')[0]
        three_level = three_level_dict[three_level_code]
        code_level[code] = three_level + [cid]

    code_level_matrix = np.zeros((len(code_map), 4), dtype=int)
    for code, cid in code_map.items():
        code_level_matrix[cid] = code_level[code]
    return code_level_matrix


def split_patients(patient_admission, admission_codes, code_map, train_num, test_num, seed=6669):
    """
    This function splits the patients into training, validation, and testing sets. The split ensure that each disease
    has at least one patient that has it in the train set
    :param patient_admission: dict, key by patient ID, value by list of admissions
    :param admission_codes: dict, key by admission ID, value by list of disease codes
    :param code_map: dict, key by disease code, value by integer
    :param train_num: int
    :param test_num: int
    :param seed: int
    :return:
    """
    np.random.seed(seed)
    common_pids = set()  # for each disease, find the first patient in patient_admission that has the disease in one of his/her admissions
    pbar = tqdm(enumerate(code_map), total=len(code_map), desc='splitting patients')
    for i, code in pbar:
        for pid, admissions in patient_admission.items():
            for admission in admissions:
                codes = admission_codes[admission[EHRParser.adm_id_col]]  # disease codes of an admission of pid
                if code in codes:
                    common_pids.add(pid)
                    break
            else:
                continue
            break

    """Find the patient with the most admissions"""
    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid

    common_pids.add(pid_max_admission_num)
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))  # the other patients
    np.random.shuffle(remaining_pids)
    valid_num = len(patient_admission) - train_num - test_num
    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist()))))
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    return train_pids, valid_pids, test_pids


def normalize_adj(adj):
    """Normalized an adjacent matrix by dividing each row by the sum of the row"""
    s = adj.sum(axis=-1, keepdims=True)
    s[s == 0] = 1
    result = adj / s
    return result


def generate_code_code_adjacent(pids, patient_admission, admission_codes_encoded, code_num, threshold=0.01, return_sim = False):
    """
    This function generates the adjacent matrix of disease codes, 2 diseases are neighbor if they appear together in at least one admission
    :param pids: list of patient IDs
    :param patient_admission: dict, key by patient ID, value by list of admissions
    :param admission_codes_encoded: dict, key by admission ID, value by list of disease codes
    :param code_num: int, number of disease codes
    :param threshold: float, threshold for the adjacent matrix
    :return:
    """
    print(termcolor.colored('generating disease code code adjacent matrix ...', 'green'))
    n = code_num
    adj = np.zeros((n, n), dtype=int)
    pbar = tqdm(enumerate(pids), total=len(pids), desc='generating disease code code adjacent matrix')
    for i, pid in pbar:
        for admission in patient_admission[pid]:
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            for row in range(len(codes) - 1):
                for col in range(row + 1, len(codes)):
                    c_i = codes[row]
                    c_j = codes[col]
                    adj[c_i, c_j] += 1
                    adj[c_j, c_i] += 1

    norm_adj = normalize_adj(adj)
    a = norm_adj < threshold # mask infrequent edges
    b = adj.sum(axis=-1, keepdims=True) > (1 / threshold)
    adj[np.logical_and(a, b)] = 0  # mask disease edges
    
    sim_matrix = torch.load('./pretraining/similarity_matrix.pt').detach().cpu().numpy()
    # sim_matrix = normalize_adj(sim_matrix)
    sim_matrix[np.logical_and(a, b)] = 0
    sim_matrix = normalize_adj(sim_matrix)

    if return_sim:
        return adj, sim_matrix
    return adj


def build_code_xy(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num):
    """
    This function generates the features and labels for each patient
    The features are the historical admissions and the labels are the diseases of the last admission
    :param pids: list of patient IDs
    :param patient_admission: dict, key by patient ID, value by list of admissions
    :param admission_codes_encoded: dict, key by admission ID, value by list of disease codes
    :param max_admission_num: int, maximum number of admissions for a patient
    :param code_num: int, number of disease codes in the dataset
    """
    n = len(pids)  # number of patient
    x = np.zeros((n, max_admission_num, code_num), dtype=bool)  # patient, admission, code
    y = np.zeros((n, code_num), dtype=int)  # patient, code
    lens = np.zeros((n,), dtype=int)
    notes = []
    for i, pid in tqdm(enumerate(pids), total=len(pids), desc="Building features and labels"):
        admissions = patient_admission[pid]
        admission_notes = []
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[
                admission[EHRParser.adm_id_col]]  # list of encoded disease codes of an admission
            x[i, k, codes] = 1
            note = admission[EHRParser.note_col]
            admission_notes.append(note)
        codes = np.array(admission_codes_encoded[admissions[-1][EHRParser.adm_id_col]])
        y[i, codes] = 1  # last admission diseases as label
        lens[i] = len(admissions) - 1
        notes.append(admission_notes)
    return x, y, lens, notes


def generate_neighbors(code_x, lens, adj):
    """
    This function generates the undiagnosed neighboring diseases of each disease code in each admission
    :param code_x: np.ndarray, shape (n, max_admission_num, code_num), features of each patient
    :param lens: np.ndarray, shape (n,), number of admissions for each patient
    :param adj: np.ndarray, shape (code_num, code_num), adjacent matrix of disease codes
    """
    n = len(code_x)
    neighbors = np.zeros_like(code_x, dtype=bool)
    for i, admissions in tqdm(enumerate(code_x), total=n, desc="Generating neighbors"): # patient i
        for j in range(lens[i]): # admission j
            codes_set = set(np.where(admissions[j] == 1)[0])  # disease index of admission j of patient i
            all_neighbors = set()
            for code in codes_set:
                code_neighbors = set(np.where(adj[code] > 0)[0]).difference(
                    codes_set)  # undiagnosed neighbors of current disease code
                all_neighbors.update(code_neighbors)
            if len(all_neighbors) > 0:
                neighbors[i, j, np.array(list(all_neighbors))] = 1

    return neighbors


def divide_middle(code_x, neighbors, lens):
    """
    This function divides the diseases of each admission into 3 categories: diagnosed in both admission j and j-1,
     diagnosed in admission j but not in j-1, and diagnosed in admission j but not in j-1 and not in the undiagnosed
     neighbors of diseases in admission j-1
    :param code_x: np.ndarray, shape (n, max_admission_num, code_num), features of each patient
    :param neighbors: np.ndarray, shape (n, max_admission_num, code_num), undiagnosed neighboring diseases of each disease code
    :param lens: np.ndarray, shape (n,), number of admissions for each patient
    """
    n = len(code_x)  # number of patients
    divided = np.zeros((*code_x.shape, 3), dtype=bool)  # n, max_admission_num, code_num, 3
    for i, admissions in tqdm(enumerate(code_x), total=n, desc="Dividing middle"): # patient i
        divided[i, 0, :, 0] = admissions[0]
        for j in range(1, lens[i]): # admission j
            codes_set = set(np.where(admissions[j] == 1)[0])  # disease index of admission j of patient i
            m_set = set(np.where(admissions[j - 1] == 1)[0])  # disease index of admission j-1 of patient i
            n_set = set(np.where(neighbors[i][j - 1] == 1)[
                            0])  # disease index of undiagnosed neighbors of diseases in admission j-1 of patient i

            m1 = codes_set.intersection(m_set)  # diseases that are diagnosed in both admission j and j-1
            m2 = codes_set.intersection(
                n_set)  # diseases that are diagnosed in admission j and in the undiagnosed neighbors of diseases in admission j-1
            m3 = codes_set.difference(m_set).difference(
                n_set)  # diseases that are diagnosed in admission j but not in admission j-1 and not in the undiagnosed neighbors of diseases in admission j-1
            if len(m1) > 0:
                divided[i, j, np.array(list(m1)), 0] = 1
            if len(m2) > 0:
                divided[i, j, np.array(list(m2)), 1] = 1
            if len(m3) > 0:
                divided[i, j, np.array(list(m3)), 2] = 1
    return divided


def build_heart_failure_y(hf_prefix, codes_y, code_map):
    """
    This function generates the heart failure binary labels for each patient
    :param hf_prefix: str, prefix of heart failure disease codes
    :param codes_y: np.ndarray, shape (n, code_num), labels of each patient
    :param code_map: dict, key by disease code, value by integer
    """
    hf_list = np.array(
        [cid for code, cid in code_map.items() if code.startswith(hf_prefix)])  # find heart disease codes
    hfs = np.zeros((len(code_map),), dtype=int)
    hfs[hf_list] = 1  # mask
    hf_exist = np.logical_and(codes_y, hfs)
    y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
    return y

def save_sparse(path, x):
    idx = np.where(x > 0)
    values = x[idx]
    np.savez(path, idx=idx, values=values, shape=x.shape)

def save_data(path, code_x, visit_lens, codes_y, hf_y, divided, neighbors, notes):
    save_sparse(os.path.join(path, 'code_x'), code_x)
    np.savez(os.path.join(path, 'visit_lens'), lens=visit_lens)
    save_sparse(os.path.join(path, 'code_y'), codes_y)
    np.savez(os.path.join(path, 'hf_y'), hf_y=hf_y)
    save_sparse(os.path.join(path, 'divided'), divided)
    save_sparse(os.path.join(path, 'neighbors'), neighbors)
    with open(os.path.join(path, 'notes.txt'), 'w', encoding='utf-8') as f:
        for note in notes:
            f.write(str(note) + '\n')

def load_sparse(path):
    data = np.load(path)
    idx, values = data['idx'], data['values']
    mat = np.zeros(data['shape'], dtype=values.dtype)
    mat[tuple(idx)] = values
    return mat

if __name__ == "__main__":
    string = "V01-V09"
    print(parse_icd9_range(string))
    print('%03d' % 10)
