# Multiple_Disease_Prediction
Description: This application predicts the likelihood of multiple chronic diseases—Liver Disease, Kidney Disease, and Parkinson’s Disease—using patient medical data such as biochemical test results, clinical indicators, and physiological measurements. By identifying early risk patterns and disease-specific markers, the system supports early diagnosis, timely medical intervention, and improved healthcare decision-making.
Technologies Used: Python, Streamlit, ML, EDA
Project Details:
Data Preparation & Pre-processing
•	Collected and analyzed separate datasets for:
o	Liver Disease
o	Kidney Disease
o	Parkinson’s Disease
•	Cleaned datasets by:
o	Handling missing and inconsistent values
o	Removing irrelevant identifiers (e.g., patient ID, name)
•	Applied:
o	Label Encoding for categorical medical features
o	Numeric type conversion for clinical attributes
o	Log Transformation to reduce skewness and manage outliers
•	Addressed class imbalance using:
o	SMOTE / SMOTETomek
•	Ensured consistent feature alignment between training and prediction pipelines.
EDA
•	Performed comprehensive EDA using Pandas and Seaborn to:
•	Understand feature distributions and correlations
•	Detect outliers and skewed medical indicators
•	Identify disease-specific risk factors
•	Analysed relationships between:
•	Lab test values and disease outcomes
•	Clinical indicators and diagnosis results
Visualization
•	Built an interactive Streamlit dashboard for HR stakeholders.
•	Visualized:
o	Disease distribution across patients
o	Boxplots highlighting abnormal clinical ranges
o	Correlation heatmaps of medical features
o	Comparative analysis of healthy vs diseased cases
Machine Learning Models
•	Trained and evaluated multiple models for Attrition Prediction and Performance Prediction:
o	Logistic Regression
o	Random Forest
o	Decision Tree
o	K-Nearest Neighbors (KNN)
o	Support Vector Machine (SVM)
o	AdaBoost
o	Gradient Boosting
o	XGBoost
Model Evaluation Metrics:
•	Accuracy
•	Precision
•	Recall
•	F1-Score
•	AUC-ROC
Model Selection & Deployment
•	Selected the best-performing model based on evaluation metrics.
•	Saved the trained model using Pickle.
•	Integrated the model into Streamlit for:
o	Liver Disease Prediction
o	Kidney Disease Prediction
o	Parkinson’s Disease Prediction
