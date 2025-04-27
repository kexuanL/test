
# MBTI classification
- [Data Processing & EDA](#data-processing--eda)
- [Logistic Regression Model](#logistic-regression-model)
- [BERT 16 + BERT 4 Classification Model](#bert-16--bert-4-classification-model)
- [RoBERTa Model](#roberta-model)
- [T5 Model](#t5-model)

- --
## Instruction for use
1. Download the files `twitter_MBTI.csv` and `mbti_1.csv`.
2. Download the following CSVs from Google Drive:
- [mbti_1.csv](https://drive.google.com/file/d/1hlGPiIc_SxBB6RQLbSV_AJsaX4NiAZwW/view?usp=drive_link)  
- [twitter_MBTI.csv](https://drive.google.com/file/d/1gp-K6AQPMj81PQ4folgcCIH4btABXh5i/view?usp=drive_link)
3. For all code file, update the file paths for `Kaggle_ds` and `tianchi_ds` in each code file to your own directories.
4. The code should then run smoothly.
---
## Data preprocessing + EDA
- Exists in each code file.
- Load the data sets / standardize the data / explode the rows / clean the text, encoding / tokenizing / embedding.
- **Notice:** We adopt length control to delete the bottom 25% of data that has short length for `MBTI_classification_BERT16_length_control`, and only use a single dataset in `MBTI_classification_single_dataset`.

---
## Logistic Regression Model
- **Logistic_Regression.ipynb**: Performs logistic regression training and evaluation for MBTI personality type classification based on pre-extracted text embeddings.
Data preprocessing 

1. Embedding 

- After preprocessing the text data, generate embeddings and save the results into embedded_df.csv. 

2. Data Preparation 

- Load the embedded_df.csv file containing embeddings and labels. 

3. Model Training: 16-Type Classification (label_16) 

- Train a logistic regression model on 16 MBTI types. 

- Evaluate performance on validation and test sets with accuracy, macro F1 score, and confusion matrices. 

4. Model Training: 4-Dimension Binary Classification (E, N, T, P) 

- Train separate logistic regression models for each binary dimension. 

- Evaluate performance with accuracy and confusion matrices.

---
## BERT 16+BERT 4 classification model
** MBTI_classification_BERT16_and_BERT_4.ipynb**: BERT-based MLP model for 16 classification and 4 classification
- **Final Model**: `bert-base-uncased` from Hugging Face + MLP
- **16-Type Classification Model**  
  - Fine-tune and select the best model  
  - Evaluate performance  
- **4-Dimension Binary Classification** (E/I, N/S, T/F, P/J)  
  - Train separate BERT-based models for each binary dimension  
  - Evaluate each binary model’s performance with accuracy and confusion matrices  
  - Evaluate the integrated model’s performance with accuracy and confusion matrices  
- **Trial Experiments**  
  - Deleting short-length posts: `MBTI_classification_BERT16_length.ipynb`  
  - Using a single dataset: `MBTI_classification_single_dataset.ipynb`  
  Both trials were conducted on the BERT-16 classification model.
---
## RoBERTa model
**RoBERTa_16.ipynb**: It utilizes a resampled dataset and leverages a multiclass classification head (num_labels=16) on top of the RoBERTa encoder. 
### Structure
- **Data Preparation**  
  Reads and preprocesses the MBTI dataset, tokenizes text inputs.

- **Model Setup**  
  Loads a pre-trained RoBERTa model and modifies the classification head for 16 classes.

- **Training Configuration**  
  Defines training parameters such as learning rate, batch size, number of epochs, evaluation strategy, and output directory.

- **Training Loop**  
  Fine-tunes the model using the `Trainer` API from Hugging Face.

- **Evaluation**  
  Evaluates the model on the validation set to monitor loss and accuracy.

- **Saving Results**  
  Stores the fine-tuned model checkpoints and tokenizer.

### Key Features

- **Tokenizer**  
  Uses `RobertaTokenizerFast` for fast and efficient tokenization.

- **Model**  
  `RobertaForSequenceClassification` adapted for multiclass (16-way) classification.

- **Training Arguments**  
  - **Learning Rate:** 2e-5  
  - **Batch Size:** 32  
  - **Epochs:** 3  
  - **Evaluation Strategy:** `epoch`  
  - **Loss & Accuracy Tracking:** Records training loss, validation loss, and validation accuracy across epochs.  
  - **Visualization:** Plots the training loss over steps to observe convergence.

### Output

- Fine-tuned RoBERTa model files (`.bin` checkpoint and tokenizer files)  
- Training loss curves  
- Accuracy metrics per epoch  
---
## T5 Model

**T5.ipynb**: Performs T5-based MBTI personality type classification using pre-extracted text embeddings.  
Training is conducted with PyTorch and Hugging Face Transformers.

---

### Data Preparation

- Load the resampled MBTI dataset  
- Split into training, validation, and test sets  
- Create:  
  - `label_16` column for the 16-type MBTI classification  
  - Binary labels for each dimension (E/I, N/S, T/F, P/J) for fine-grained evaluation  

---

### Dataset Construction

- Build a `DatasetDict` with Hugging Face Datasets  
- **Inputs:** Prompt templates asking the model to predict MBTI type  
- **Targets:** Corresponding MBTI labels  

---

### Model Setup

- Load pre-trained `t5-base` (with `dropout=0.1`)  
- Load the corresponding T5 tokenizer  
- Adjust configuration:  
  - `batch_size = 16`  
  - `num_train_epochs = 3`  

---

### Preprocessing

- Tokenize input prompts and target labels  
- Apply appropriate `max_length` and padding strategies for both source and target sequences  

---

### Training Configuration

- Define `TrainingArguments` including:  
  - `output_dir`  
  - `per_device_train_batch_size = 16`  
  - `num_train_epochs = 3`  
  - `evaluation_strategy = "epoch"`  
  - Mixed precision (if enabled)  
  - Logging setup (e.g., logging steps, save strategy)  

---

### Training Loop

- Use `Trainer` API to fine-tune T5  
- Perform real-time evaluation on the validation set after each epoch  

---

### Evaluation

- Monitor and record:  
  - Training loss  
  - Validation loss  
  - Validation accuracy  
- Review metrics after each epoch to assess convergence and model performance  













