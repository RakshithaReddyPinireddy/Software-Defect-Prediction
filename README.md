# Software Defect Prediction Using an Intelligent Ensemble-Based Model

 ## Project Overview
Software Defect Prediction aims to identify defective modules in software systems at an early stage of development. Early detection helps reduce cost, improve software quality, and support better decision-making during testing and maintenance.

This project uses **machine learning ensemble techniques** to predict software defects using historical project data.

---

##  Objectives
- Predict defective and non-defective software modules
- Improve prediction accuracy using ensemble learning
- Reduce testing effort and maintenance cost
- Analyze and compare model performance

---

##  Technologies Used
- Programming Language: **Python**
- Libraries:
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn
- Development Environment: **VS Code**
- Version Control: **Git & GitHub**

---

##  Machine Learning Models
The following models are used in the ensemble:
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes
- Artificial Neural Network (ANN)

The final prediction is obtained by combining the outputs of multiple models to improve accuracy and reliability.

---

##  Dataset
- **NASA MDP (Metrics Data Program) Dataset**
- Contains software metrics and defect labels
- Widely used benchmark dataset for defect prediction research

---

##  Methodology
1. Data collection and preprocessing  
2. Feature selection and normalization  
3. Training individual machine learning models  
4. Ensemble model creation  
5. Performance evaluation using metrics  

---

##  Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

##  Results
- Ensemble model achieved better performance compared to individual models
- Improved defect detection accuracy
- Reduced false positives and false negatives

---

## Project Structure
software-defect-prediction/
│
├── dataset/
├── models/
├── preprocessing/
├── results/
├── main.py
├── README.md
└── .gitignore

## How to Run
1.Clone the repository: git clone https://github.com/your-username/your-repo-name.git
2.Navigate to the project directory: cd software-defect-prediction
3.Install required libraries: pip install -r requirements.txt
4.Run the main file: python main.py

