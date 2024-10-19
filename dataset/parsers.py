import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import termcolor
from tqdm import tqdm


class EHRParser:
    """
    Utility class to parse raw EHR data into usable format
    """
    pid_col = 'pid'  # patient ID
    adm_id_col = 'adm_id'  # admission ID
    adm_time_col = 'adm_time'  # admission time
    cid_col = 'cid'  # code ID
    note_col = 'note' # note text
    note_category_col = 'category' # note category

    def __init__(self, path: str):
        self.path = path  # path to raw data folder
        self.skip_pid_check = False
        self.patient_admission = None  # dict, key by patient ID, value by list of admissions
        self.admission_codes = None  # dict, key by admission ID, value by list of disease codes
        self.admission_procedures = None
        self.admission_medications = None

        self.parse_fn = {'d': self.set_diagnosis}

    def set_diagnosis(self):
        raise NotImplementedError

    def set_admission(self):
        raise NotImplementedError

    def set_notes(self):
        raise NotImplementedError

    def _after_read_admission(self, admissions, cols) -> pd.DataFrame:
        """
        This function is for admissions post-processing
        :param admissions: pd.DataFrame
        :param cols: dict of strings
        :return: pd.DataFrame
        """
        return admissions

    def _after_read_concepts(self, concepts, concept_type, cols) -> pd.DataFrame:
        """
        This function is for concepts post-processing
        :param concepts: pd.DataFrame
        :param concept_type: str
        :param cols: dict of strings
        :return: pd.DataFrame
        """
        return concepts

    def parse(self, sample_num=None, seed=6669):
        self.parse_admission()
        self.parse_diagnoses()
        self.parse_notes()

        self.calibrate_patient_by_admission()
        self.calibrate_admission_by_patient()
        # self.calibrate_patient_by_notes()
        if sample_num is not None:
            self.sample_patients(sample_num, seed)
        return self.patient_admission, self.admission_codes

    def parse_admission(self):
        """
        Function to parse admission from csv file
        :return:
        """
        print(termcolor.colored("Parsing admission csv file ...", 'green'))
        filename, cols, converters = self.set_admission()

        """Read admission csv file"""
        admissions = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        """Post processing admissions"""
        admissions = self._after_read_admission(admissions, cols)
        all_patients = OrderedDict()  # dict, key by patient ID, value by list of admissions
        pbar = tqdm(admissions.iterrows(), total=len(admissions), desc="Parsing admissions")
        for i, row in pbar:
            """Get patient ID, admission ID, and admission time"""
            pid, adm_id, adm_time = row[cols[self.pid_col]], row[cols[self.adm_id_col]], row[cols[self.adm_time_col]]
            if pid not in all_patients:
                all_patients[pid] = []
            all_patients[pid].append({self.adm_id_col: adm_id, self.adm_time_col: adm_time})

        patient_admission = OrderedDict()
        """Filter, keep only patients with at least 2 admissions"""
        for pid, admission_list in all_patients.items():
            if len(admission_list) >= 2:  # patient with at least 2 admissions
                patient_admission[pid] = sorted(admission_list, key=lambda x: x[self.adm_time_col])
        self.patient_admission = patient_admission

    def parse_diagnoses(self):
        """
        Function to parse diagnosis from csv file
        :return:
        """
        print(termcolor.colored("Parsing diagnosis csv file ...", 'green'))
        self.admission_codes = self._parse_concept('d')  # dict, key by admission ID, value by list of disease codes

    def parse_notes(self, use_summary=False):
        """
        Function to parse notes from csv file, add it as an attribute of admission in patient_admission
        :return: patient_admission
        """
        print(termcolor.colored("Parsing notes csv file ...", 'green'))
        filename, cols, converters = self.set_notes()
        notes = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        pbar = tqdm(self.patient_admission.items(), total=len(self.patient_admission), desc="Parsing notes")
        count_found = 0
        count_not_found = 0
        for pid, admissions in pbar:
            # select from notes
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if use_summary:
                    selected_notes = notes[(notes[cols[self.pid_col]] == pid) & (notes[cols[self.adm_id_col]] == adm_id)
                                           & (notes[cols[self.note_category_col]] == 'Discharge summary')][
                        cols[self.note_col]]
                else:
                    selected_notes = notes[(notes[cols[self.pid_col]] == pid) & (notes[cols[self.adm_id_col]] == adm_id)
                                           & (notes[cols[self.note_category_col]] != 'Discharge summary')][
                        cols[self.note_col]]
                text_note = '. '.join(selected_notes)
                if len(text_note) > 0:
                    count_found += 1
                else:
                    count_not_found += 1
                admission[self.note_col] = text_note
                pbar.set_postfix({'found': count_found, 'not found': count_not_found})
        return self.patient_admission

    def _parse_concept(self, concept_type):
        """
        Function to parse general concept from csv file
        :param concept_type:
        :return:
        """
        assert concept_type in self.parse_fn.keys()
        filename, cols, converters = self.parse_fn[concept_type]()
        """Read concepts csv file"""
        concepts = pd.read_csv(os.path.join(self.path, filename), usecols=list(cols.values()), converters=converters)
        concepts = self._after_read_concepts(concepts, concept_type, cols)
        results = OrderedDict()  # dict, key by admission ID, value by list of codes
        pbar = tqdm(concepts.iterrows(), total=len(concepts), desc="Parsing %s" % concept_type)
        for i, row in pbar:
            pid = row[cols[self.pid_col]]
            if self.skip_pid_check or pid in self.patient_admission:
                adm_id, code = row[cols[self.adm_id_col]], row[cols[self.cid_col]]
                if code == '':  # found an unlabeled concept
                    continue
                if adm_id not in results:
                    results[adm_id] = []
                results[adm_id].append(code)
        return results

    def calibrate_patient_by_admission(self):
        """
        This function deletes patients with an admission that is not recorded in self.admission_codes
        :return:
        """
        print(termcolor.colored("Calibrating patient by admission ...", 'green'))
        del_pids = []
        for pid, admissions in self.patient_admission.items():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                if adm_id not in self.admission_codes:
                    break
            else:
                continue
            del_pids.append(pid)

        # delete patients with an admission that is not recorded in self.admission_codes
        for pid in del_pids:
            admissions = self.patient_admission[pid]
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                for concepts in [self.admission_codes]:
                    if adm_id in concepts:
                        del concepts[adm_id]

            del self.patient_admission[pid]

    def calibrate_admission_by_patient(self):
        """
        This function deletes admissions that are not recorded in self.patient_admission
        :return:
        """
        print(termcolor.colored("Calibrating admission by patient ...", 'green'))
        adm_id_set = set()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id_set.add(admission[self.adm_id_col])
        del_adm_ids = [adm_id for adm_id in self.admission_codes if adm_id not in adm_id_set]
        for adm_id in del_adm_ids:
            del self.admission_codes[adm_id]

    def sample_patients(self, sample_num, seed=6669):
        """
        Function to sample patients
        """
        np.random.seed(seed)
        keys = list(self.patient_admission.keys())
        selected_pids = np.random.choice(keys, sample_num, False)
        self.patient_admission = {pid: self.patient_admission[pid] for pid in selected_pids}
        admission_codes = dict()
        for admissions in self.patient_admission.values():
            for admission in admissions:
                adm_id = admission[self.adm_id_col]
                admission_codes[adm_id] = self.admission_codes[adm_id]
        self.admission_codes = admission_codes


class Mimic3Parser(EHRParser):
    def set_admission(self):
        filename = 'ADMISSIONS.csv.gz'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.adm_time_col: 'ADMITTIME'}
        converters = {
            'SUBJECT_ID': int,
            'HADM_ID': int,
            'ADMITTIME': lambda cell: datetime.strptime(str(cell), '%Y-%m-%d %H:%M:%S')
        }
        return filename, cols, converters

    def set_diagnosis(self):
        filename = 'DIAGNOSES_ICD.csv.gz'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.cid_col: 'ICD9_CODE'}
        converters = {
            'SUBJECT_ID': int,
            'HADM_ID': int,
            'ICD9_CODE': Mimic3Parser.to_standard_icd9
        }
        return filename, cols, converters

    def set_notes(self):
        filename = 'NOTEEVENTS.csv.gz'
        cols = {self.pid_col: 'SUBJECT_ID', self.adm_id_col: 'HADM_ID', self.note_col: 'TEXT',
                self.note_category_col: 'CATEGORY'}
        converters = {
            'SUBJECT_ID': int,
            'HADM_ID': lambda x: int(x) if x != '' else -1,
            'TEXT': str,
            'CATEGORY': str
        }
        return filename, cols, converters

    @staticmethod
    def to_standard_icd9(code: str):
        code = str(code)
        if code == '':
            return code
        split_pos = 4 if code.startswith('E') else 3
        icd9_code = code[:split_pos] + '.' + code[split_pos:] if len(code) > split_pos else code
        return icd9_code
