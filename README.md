# FakeNews_Dissertation-

# Author Name : Govardhan Cherukuri 
# Student Id : 34048444

# Fake News Classification Using NLP, Machine Learning and Deep Learning

## 1. Project Overview

This project presents the development and evaluation of a text-based fake news classification framework using Natural Language Processing (NLP), Machine Learning (ML), and Deep Learning (DL) techniques.

The workflow covers dataset collection, exploratory data analysis, text preprocessing, TF-IDF and Bag of Words feature representation, baseline modelling, GridSearchCV hyperparameter optimisation, five-fold cross-validation, model comparison, SHAP analysis, prototype development, and human evaluation.

The final selected model was tuned Logistic Regression using Bag of Words (BoW), which achieved the highest F1-Score among the evaluated experimental configurations.

## 2. Project Aim

The aim of this dissertation is to develop and evaluate a machine learning and deep learning-based fake news classification framework using NLP techniques and prototype development.

## 3. Objectives

1. Investigate and preprocess a publicly available labelled fake news dataset.
2. Apply NLP preprocessing techniques to prepare textual news content.
3. Implement and compare TF-IDF and Bag of Words representations.
4. Develop Logistic Regression, Random Forest, ANN and CNN classification models.
5. Evaluate models using Accuracy, Precision, Recall, F1-Score and ROC-AUC.
6. Apply GridSearchCV for hyperparameter optimisation.
7. Validate selected models using five-fold cross-validation.
8. Select the best-performing model primarily using F1-Score.
9. Apply SHAP analysis to examine influential textual features.
10. Develop a functional fake news classification prototype.
11. Conduct structured human evaluation of the developed prototype.

## 4. Dataset

The project uses the WELFake dataset obtained through the Kaggle dataset published by Saurabh Shahane.

Dataset link:

https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

The dataset contains accessible news articles, including  real and fake news articles. The task is binary classification, where 0 represents fake news and 1 represents real news.

Dataset citation:

Shahane, S. (2023). *Fake news classification* [Dataset]. Kaggle.

## 5. Technologies Used

### Programming Language

- Python

### Main Libraries

- pandas
- NumPy
- Matplotlib
- NLTK
- scikit-learn
- SHAP
- TensorFlow
- Keras

## 6. Text Preprocessing

The textual data was processed before feature extraction and model training.

The preprocessing workflow included:

1. Converting text to lowercase.
2. Removing URLs.
3. Removing HTML tags.
4. Removing numbers and punctuation.
5. Removing unnecessary whitespace.
6. Tokenising the cleaned text.
7. Removing English stopwords.
8. Lemmatizing words using WordNet.

The purpose was to reduce unnecessary textual variation and create a consistent representation before numerical feature extraction.

## 7. Exploratory Data Analysis

EDA was conducted to understand the structure and characteristics of the dataset.

The analysis included:

- Dataset dimensions
- Column inspection
- Data types
- Missing-value investigation
- Class distribution
- Text-length analysis
- Word-frequency analysis
- Distribution visualisations

## 8. Feature Representation

### 8.1 TF-IDF

Term Frequency-Inverse Document Frequency was used to transform news text into numerical feature vectors. TF-IDF assigns greater importance to informative terms while reducing the influence of terms occurring frequently throughout the corpus.

### 8.2 Bag of Words

Bag of Words represents documents using word-frequency information. Unlike TF-IDF, BoW does not apply inverse-document-frequency weighting.

Both representations were implemented to determine which provided stronger classification performance.

## 9. Classification Models

### 9.1 Logistic Regression

Logistic Regression was included as a traditional binary classification baseline suitable for high-dimensional sparse textual representations.

### 9.2 Random Forest

Random Forest was included as a tree-based ensemble approach, providing a non-linear ML comparison.

### 9.3 Artificial Neural Network

ANN was implemented to evaluate a neural-network-based approach and non-linear learning capability.

### 9.4 Convolutional Neural Network

CNN was included to investigate whether convolution-based feature learning could capture useful textual patterns.

## 10. Experimental Workflow

```text
Dataset Collection
        |
        v
Data Cleaning
        |
        v
Exploratory Data Analysis
        |
        v
Text Preprocessing
        |
        +------------------+
        |                  |
        v                  v
      TF-IDF              BoW
        |                  |
        +--------+---------+
                 |
                 v
       Train/Test Splitting
                 |
                 v
       Baseline ML/DL Models
                 |
                 v
        Performance Evaluation
                 |
                 v
       GridSearchCV Optimisation
                 |
                 v
       Optimised ML/DL Models
                 |
                 v
       Five-Fold Cross-Validation
                 |
                 v
       Best Model Selection
                 |
                 v
            SHAP Analysis
                 |
                 v
       Prototype Development
                 |
                 v
        Human Evaluation
```

## 11. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

F1-Score was used as the principal criterion for final model selection because it balances Precision and Recall.

## 12. Hyperparameter Optimisation

GridSearchCV was used to systematically search predefined parameter combinations.

The optimisation process was:

```text
Model
  |
  v
Define Parameter Grid
  |
  v
GridSearchCV
  |
  v
Cross-Validated Parameter Evaluation
  |
  v
Best Parameter Combination
  |
  v
Refitted Optimised Model
  |
  v
Test-Set Evaluation
```

## 13. Five-Fold Cross-Validation

Five-fold cross-validation was conducted to examine consistency across different data partitions. The dataset was divided into five folds, with four folds used for training and one for validation. The process was repeated until every fold had served as the validation set.

## 14. Baseline TF-IDF Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 92.21% | 92.17% | 92.15% | 92.14% | 97.91% |
| Random Forest | 93.17% | 93.15% | 93.05% | 93.08% | 98.37% |
| ANN | 92.95% | 91.27% | 94.27% | 92.74% | 97.88% |
| CNN | 91.90% | 91.80% | 91.78% | 91.78% | 97.29% |

## 15. Baseline BoW Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 93.24% | 93.06% | 93.36% | 93.16% | 98.12% |
| Random Forest | 93.21% | 93.13% | 93.03% | 93.07% | 98.28% |
| ANN | 93.10% | 94.13% | 91.96% | 93.03% | 97.79% |
| CNN | 92.83% | 92.71% | 92.58% | 92.63% | 97.91% |

## 16. Optimised TF-IDF Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 93.38% | 93.53% | 93.01% | 93.27% | 98.04% |
| Random Forest | 93.21% | 93.14% | 93.08% | 93.11% | 98.38% |
| ANN | 92.59% | 91.08% | 94.20% | 92.61% | 97.82% |
| CNN | 91.45% | 92.58% | 89.86% | 91.20% | 97.29% |

## 17. Optimised BoW Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 93.76% | 93.22% | 94.20% | **93.70%** | 97.47% |
| Random Forest | 93.59% | 93.62% | 93.36% | 93.49% | 98.33% |
| ANN | 93.28% | 93.70% | 92.59% | 93.14% | 97.86% |
| CNN | 93.48% | 92.47% | 94.48% | 93.46% | **98.40%** |

## 18. Best Performing Model

The final model selected for the prototype was:

**Tuned Logistic Regression + Bag of Words**

| Metric | Result |
|---|---:|
| Accuracy | 93.76% |
| Precision | 93.22% |
| Recall | 94.20% |
| F1-Score | **93.70%** |
| ROC-AUC | 97.47% |

The model was selected primarily according to F1-Score because it achieved the highest F1-Score across the evaluated experimental configurations.

## 19. Cross-Validation Results

| Model | Representation | Mean Accuracy | Standard Deviation |
|---|---|---:|---:|
| Logistic Regression | TF-IDF | 93.40% | 0.54% |
| Logistic Regression | BoW | **93.71%** | 0.54% |
| Random Forest | TF-IDF | 93.40% | 0.46% |
| Random Forest | BoW | 93.62% | 0.54% |

## 20. SHAP Analysis

SHAP analysis was applied to the selected Logistic Regression model to examine influential textual features.

Important features identified included:

- `reuters`
- `said`
- `via`
- `hillary`
- `video`
- `obama`
- `breitbart`
- `trump`
- `twitter`

The analysis demonstrated that feature contributions can differ according to individual predictions and their relationship with other learned textual patterns.

## 21. Prototype

A functional fake news classification prototype was developed using the selected model.

The workflow is:

```text
User enters news article
        |
        v
Text preprocessing
        |
        v
BoW transformation
        |
        v
Tuned Logistic Regression
        |
        v
Prediction
        |
        v
Fake News / Real News
```

The prototype provides:

- News article input
- Prediction function
- Fake/real classification
- Prediction confidence
- Prediction dashboard
- Feedback functionality
- Reset functionality

## 22. Human Evaluation

A structured human evaluation involved 100 participants and seven predefined factors:

1. Prediction Accuracy
2. Output Clarity
3. Ease of Use
4. Usefulness
5. Confidence
6. Responsiveness
7. Satisfaction

An overall rating and open comments were also collected.

The reported results showed a positive response, with the seven main factors achieving a mean rating of 4.89/5 and the overall rating achieving 4.90/5.

## 23. Key Findings

1. TF-IDF and BoW were both effective textual representations.
2. BoW provided the strongest overall experimental condition.
3. Logistic Regression performed particularly strongly with BoW.
4. GridSearchCV provided systematic hyperparameter optimisation.
5. Five-fold cross-validation showed consistent performance.
6. Tuned BoW Logistic Regression achieved the highest F1-Score.
7. The final F1-Score was 93.70%.
8. SHAP identified influential textual features.
9. A functional prototype was developed.
10. Human evaluation indicated strong perceived usability and acceptance.

## 24. Limitations

The project has several limitations:

- The study uses one primary public dataset.
- The classification task is binary.
- The system focuses on textual news.
- It does not perform continuous real-time monitoring.
- It does not automatically verify claims against external fact-checking databases.
- It does not perform multilingual classification.
- It does not analyse image, video or audio misinformation.
- Generalisability to unseen sources and future news environments requires further investigation.

## 25. Future Work

Potential future development includes:

- BERT-based classification
- RoBERTa and DeBERTa
- Large Language Models
- Transformer fine-tuning
- Multimodal fake news detection
- Image-text fusion
- Continual learning
- Domain adaptation
- Cross-source evaluation
- Ensemble and stacking models
- Retrieval-augmented fact verification
- Real-time news monitoring
- Multilingual classification
- External knowledge-base integration



## 26. Installation

Install the required Python packages:

```bash
pip install pandas numpy matplotlib nltk scikit-learn shap tensorflow
```

Download the required NLTK resources:

```python
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## 27. Running the Project
```text
1. Load Dataset
2. Inspect Dataset
3. Perform EDA
4. Clean Text
5. Generate TF-IDF Features
6. Generate BoW Features
7. Split Dataset
8. Train Baseline Models
9. Evaluate Baseline Models
10. Run GridSearchCV
11. Evaluate Optimised Models
12. Perform Five-Fold Cross-Validation
13. Select Best Model
14. Run SHAP Analysis
15. Save Model and Vectorizer
16. Launch Prototype
```

## 28. Reproducibility

For reproducibility:

- Use the same dataset version.
- Maintain consistent preprocessing.
- Use fixed random states where applicable.
- Maintain the same train-test split.
- Use the same feature-extraction settings.
- Record GridSearchCV parameter grids.
- Save the trained model.
- Save the fitted BoW vectorizer.
- Record evaluation metrics.
- Preserve SHAP analysis outputs.

## 29. Academic Context

This repository supports the dissertation:

**Evaluating Machine Learning Models for Fake News Classification Using Natural Language Processing Techniques and Prototype Development**

The implementation covers the complete workflow from dataset preparation through model evaluation and prototype development.

## 30. Final Result

The principal result of the project is:

**Tuned Logistic Regression + Bag of Words**

**F1-Score: 93.70%**

This configuration achieved the highest F1-Score among the evaluated experimental conditions and was selected as the final model for prototype development and subsequent analysis.


Shahane, S. (2023). *Fake news classification* [Dataset]. Kaggle.
