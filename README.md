# PROJECT TITLE
Building an AI-Based Ensemble System for Corporate Bankruptcy Prediction: A Case Study of US-Listed Companies
MSc Data Analytics with Banking and Finance Research Project. This repository contains the full code for a stacking ensemble that predicts corporate bankruptcy from company financial figures, together with a working web prototype

# What this project does
Three different machine learning models (Random Forest, Gradient Boosting and k-Nearest Neighbours) are each trained to predict bankruptcy. A fourth model, a "meta-learner", then learns how much to trust each of the three before giving a final answer. This combined approach is called a stacking ensemble. The system is evaluated honestly: it is tested only on company data from years the models have never seen (2015-2018), with the natural imbalance between healthy and bankrupt companies left untouched.

# Repository contents
- EDA.ipynb: Explores the raw data and produces five charts, including a correlation heatmap and a health and bankrupt comparison
- Train_pipeline.ipynb: Prepares the data, trains all four models, evaluates results and saves the trained pipeline
- Streamlit_ Interface:  The web prototype users interact with (streamlit runs this)
- Requirements.txt: All python packages needed to run the project
- The _bankruptcy _data.csv: See dataset below
- Trained_pipeline_final. joblib: The saved trained models, created by running step 2
- 
# How to run it
1. Install the required packages
pip install -r requirements.txt

2. Explore the data (optional)
Jupyter notebook step1_eda.ipynb

3. Train the models. This must be run before Step 4, as it creates the
saved model file the web app depends on. Jupyter notebook step2_train_pipeline. ipynb
This prints the full results and saves `trained_pipeline_final.joblib.

4. Launch the web prototype: streamlit run streamlit_interface.py
A page will open in your browser automatically.

# Results
Evaluated on 12,282 unseen company-year records (2015-2018):
# At Threshold 0.40
Model                AUC     Macro-F1    Type I    Type II
Random Forest        0.7401   0.2929       0.6353    0.1115
Gradient Boosting    0.7472   0.1995       0.7910    0.0592
k-Nearest Neighbours 0.7373   0.3964       0.4419    0.2021
Stacking ensemble    0.7740   0.5664       0.0530    0.7038
 Model 1: Random Forest
AUC =   0.7401
Macro-F1 =0.2929
Type I =  0.6353
Type II = 0.1115
# At Threshold 0.30
Model                AUC     Macro-F1    Type I    Type II
Random Forest        0.7401   0.2274       0.7514   0.0488
Gradient Boosting    0.7472   0.1454       0.8634   0.0383
k-Nearest Neighbours 0.7373   0.3964       0.4419    0.2021
Stacking ensemble    0.7740   0.5311       0.1156   0.5923

The ensemble outperforms every individual model on Macro-F1, the metric most
appropriate for imbalanced data. Note that Gradient Boosting's individual
score collapses under cost weighting, yet the ensemble does not inherit this
weakness, evidence that the meta-learner correctly learns to trust it less.

Two decision thresholds are reported: 0.40 (best overall balance) and
0.30 (catches more real bankruptcies, at the cost of more false alarms).
Both were selected using validation data only, never the test data.

# Data Source
The bankruptcy prediction dataset used in this project is the Bankruptcy Prediction Dataset related to American companies in the stock market (1999–2018), made publicly available by Sowide on GitHub: https://github.com/sowide/bankruptcy_dataset and available on Kaggle https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset. 
The dataset contains financial information for US-listed companies over the 1999–2018 period and is used to develop and evaluate the machine-learning models presented in this project.

# Dataset License
The dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0) license. Under this license, the dataset may be shared, reproduced, and adapted, provided that appropriate attribution is given to the original creator/source and any modifications are indicated.

# Attribution
Dataset source:
Sowide, Bankruptcy prediction dataset related to the American companies in the stock market (1999–2018). GitHub repository. Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
Source: https://github.com/sowide/bankruptcy_dataset and available on Kaggle athttps://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset

This project uses the dataset for academic and research purposes as part of an MSc dissertation on AI-based ensemble modelling for corporate bankruptcy prediction.

Use and Modification
The dataset was used as the input data for the development and evaluation of the proposed bankruptcy prediction system. Data preprocessing, feature selection, model training, ensemble construction, and evaluation were performed as part of this research. The original dataset has not been presented as an original creation of this project. Any modifications or processing applied to the dataset for modelling purposes are described in the accompanying research methodology.


# Link to the prototype
[Bankruptcy Risk Prototype · Streamlit](https://bankruptcy-risk-prototype-2026project.streamlit.app/)

Author
Gabriel Boateng





- 
