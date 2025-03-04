# MDKG
## The MDKG Construction Pipeline

![gs0622B](https://github.com/user-attachments/assets/b4b94e2b-cf76-4751-a474-a5f9d9f32529)

This repository contains codes related to the MDKG (Medical Knowledge Graph) project.

### 1. NER & RE Model Training and Prediction Code + Active Learning Code

The NER&RE_model directory contains a fine-tuned joint Named Entity Recognition (NER) and Relation Extraction (RE) model based on the [Spert.PL](https://github.com/your-repo/spert.pl) framework.

## Setup

### a. Download Pretrained Models
- Download the [CODER++](https://huggingface.co/GanjinZero/coder_eng_pp) model or other BERT embedding models
- Place the downloaded model in:  
  `NER&RE_model/InputsAndOutputs/pretrained/`

### b. Prepare Dataset
- Download annotated datasets from:  
  [Zenodo Repository](https://zenodo.org/records/10960357)
- Place the data files in:  
  `NER&RE_model/InputsAndOutputs/data/dataset/`

### c. Data Preprocessing
   ​**Generate base input data**:
   ```bash
   python NER&RE_model/SynSpERT/generate_input.py
  ```
  **Generate Augmented Data (optional)**：
  ```bash
  python NER&RE_model/SynSpERT/generate_augmented_input.py'
  ```

### d. Model Training
   **Run the main training script**:
   ```bash
   python NER&RE_model/SynSpERT/main.py
   ```

### 2. `Active_learning.py` provides an active learning strategy based on [ACTUNE](https://github.com/your-repo/actune) for abstract selection.

### 3. Extract Table Text from Full PDF Files

- The `table_extraction.py` script extracts tables containing **baseline characteristics** from PDFs.
- The `query_for_table_information_extraction.py` script extracts **study population characteristics**.

### 3. Entity Linking

- The `Entity_linking.py` script links entities to the following biomedical ontologies:
  - **HPO** (Human Phenotype Ontology)
  - **GO** (Gene Ontology)
  - **UBERON** (Anatomical Ontology)
  - **MONDO** (Disease Ontology)
  - **UMLS** (Unified Medical Language System)



