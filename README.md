# Capstone project for BK-SOICT-MLwithGraphs-Fall2024
# Topic: Temporal Health Event Prediction with Dynamic Disease Graphs Modelling

## Team members: 
### 1. Nguyen Minh Chau - 20241007M - chau.nm241007M@sis.hust.edu.vn
### 2. Nguyen Tuan Dung - 20232198M - dung.nt232198M@sis.hust.edu.vn

## Project description:
This project is about predicting the health events of a patient based on the diagnosis history. A diagnosis history is a temporal sequence of admissions, each comprised of several diseases.
A disease graph is constructed based on the co-occurrences of diseases in the admission. Then, the diagnosis history can be represented as a series of subgraphs, each corresponding to an admission. The prediction task is to predict the next admission of a patient based on the previous admissions.
This method is based on the paper "Context-aware Health Event Prediction via Transition Functions on Dynamic
Disease Graphs" by Chu et al. (2021). [link](https://arxiv.org/abs/2112.05195)

The original method deploy a basic Graph Convolution for graph aggregation. In this project, we explore another graph layer with attention mechanism (GAT) as well as initialized node embeddings with node2vec and an external text embeddings to improve the final prediction.

## Dataset and tasks:
The dataset used in this project is the MIMIC-III dataset, which contains a large number of electronic health records of more than 7000 patients, each with more than 1 admissions. The dataset can be accessed at [link](https://physionet.org/content/mimiciii/1.4/).
The MIMIC-III dataset has been used as a benchmark for health event predictions tasks. Two common tasks are heart disease prediction and multiple disease prediction. 
In this project, we will report the results of both tasks with different models.

## Getting started:
1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Download the MIMIC-III dataset (the ``mimiciii/1.4`` folder) from https://physionet.org/content/mimiciii/1.4/ and extract it. You should see some .csv files
4. Run the preprocessing script: `python preprocess.py --raw_path path/to/mimiciii/1.4`. After that, the `./data` folder in the repository will looks like this:
```
./data
├── encoded
│   ├── code_map.pkl
│   ├── codes_encoded.pkl
│   ├── patient_admissions.pkl
│   ├── pids.pkl
├──parsed
│   ├── admission_codes.pkl
│   ├── code_levels.pkl
│   ├── patient_admission.pkl
├──standard
│   ├── train
│   │   ├── code_x.npz
│   │   ├── code_y.npz
│   │   ├── divided.npz
│   │   ├── hf_y.npz
│   │   ├── neighbors.npz
│   │   ├── notes.txt
│   │   ├── visit_lens.npz
│   ├── valid
│   ├── test
icd9.txt
```
5. Run the training script with the desired config file, see the example below:
```
python train.py --config configs/chet_h.json
```
You can visit the `configs` folder to see the available config files.

6. Run the evaluation script with the desired config file, see the example below:
```
python evaluate.py --config configs/chet_h.json
```
7. Optionally, you can run these scripts to generate the pretrained node embeddings before you train the models:
   - Generate node2vec embeddings: 
    ```
    cd pretraining/
    python node2vec_train.py --config path-to-chet-config.yaml
    ```
    The generated node2vec embedding should appear in`./pretraining/embeddings_node2vec.pt`.
   - Generate text embeddings:
    ```
    cd pretraining/
    python initialize_embeddings.py
    python convert.py
    ```
    The generated text embedding should appear in `./pretraining/bge_embeddings.pt`.
    
    Then you can replace the ``pretrained_embeddings_path`` field in the config file with the path to the generated embeddings to use them in the training process. 