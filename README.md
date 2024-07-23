# MDKG
## The MDKG Construction Pipeline

![gs0622B](https://github.com/user-attachments/assets/b4b94e2b-cf76-4751-a474-a5f9d9f32529)

This repository contains codes related to the MDKG (Medical Knowledge Graph) project.

### 1. NER & RE Model Training and Prediction Code + Active Learning Code

In our `NER&RE_model` file, we provide a fine-tuned joint entity recognition and relationship extraction model based on the [Spert.PL](https://github.com/your-repo/spert.pl) framework, integrated with an active learning strategy based on [ACTUNE](https://github.com/your-repo/actune).

For the training process, you can download the [CODER++](https://huggingface.co/GanjinZero/coder_eng_pp) embedding model.

### 2. Extract Table Text from Full PDF/XML Files

The `Contextual_features.py` script provides code for processing PDF/XML files, detecting tables, extracting text, and using ChatGPT to extract study population characteristics.

---

### NER & RE Model Training and Prediction

This section includes:
- Training a joint entity recognition and relationship extraction model.
- Integrating an active learning strategy to improve model performance iteratively.

#### Steps:
1. Download the CODER++ embedding model from [Hugging Face](https://huggingface.co/GanjinZero/coder_eng_pp).
2. Train the model using the provided scripts in the `NER&RE_model` directory.
3. Use the active learning code to iteratively improve the model by selecting the most informative samples for labeling.

### Extract Table Text from PDF/XML Files

This section includes:
- Processing PDF/XML files to detect tables.
- Extracting text from detected tables.
- Using ChatGPT to extract and verify study population characteristics.

