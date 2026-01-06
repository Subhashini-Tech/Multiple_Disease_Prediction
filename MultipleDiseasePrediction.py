import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay
)
from sklearn.metrics import classification_report
from imblearn.combine import SMOTETomek
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from scipy import stats
import base64
import pickle
import warnings
warnings.filterwarnings("ignore")

#### *******************LIVER DISEASE PREDICTION***************************
# Fetch the liver data
data = pd.read_csv(r"D:\Project4\indian_liver_patient.csv")
data.head(5)

#  To check null values
data.isnull().sum()

# Handling missing data
data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].median())

# New data
new_data = data
new_data.head(5)

# Label encoding
new_data['Gender'] = np.where(new_data['Gender']=='Male', 1,0)
#new_data.shape

# Outlier Check
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12,10))
axes = axes.flatten()

for ax, col in zip(axes,new_data.columns):
    sns.boxplot(new_data[col], ax=ax)

for ax in axes[len(new_data.columns):]:
    ax.set_visible(False)

plt.show()
plt.tight_layout()

# Outlier features
outlier_cols = []

for col in new_data.columns:
    Q1 = new_data[col].quantile(0.25)
    Q3 = new_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # check if any values are outside bounds
    if ((new_data[col] < lower) | (new_data[col] > upper)).any():
        outlier_cols.append(col)

#outlier_cols

# Handling Outliers - log, log2, log10, inverse, sqrt Transformation
new_data['Total_Bilirubin'] = np.log2(new_data['Albumin_and_Globulin_Ratio'])
new_data['Direct_Bilirubin'] = np.log2(new_data['Direct_Bilirubin'])
new_data['Total_Protiens'] = np.log2(new_data['Total_Protiens'])
new_data['Aspartate_Aminotransferase'] = np.log10(new_data['Aspartate_Aminotransferase'])
new_data['Albumin_and_Globulin_Ratio'] = np.log(new_data['Albumin_and_Globulin_Ratio'])
new_data['Alamine_Aminotransferase'] = np.sqrt(new_data['Alamine_Aminotransferase'])
new_data['Alkaline_Phosphotase'] =  1/new_data['Alkaline_Phosphotase']

# Features and Target
X = new_data.drop(['Dataset'], axis=1)
y = new_data['Dataset']

# SMOTE Technique
smote = SMOTETomek()
X_smote, y_smote = smote.fit_resample(X,y)

# Split Test and Train
X_train,X_test,y_train,y_test = train_test_split(X_smote,y_smote, test_size=0.3, random_state=33)

# pickle file loading
@st.cache_resource
def load_liver_model():
    with open("D:/Project4/liver_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

liver_model = load_liver_model()
print(type(liver_model))

# ***************************KIDNEY DISEASE PREDICTION************************************
# Fetch the data
kdata = pd.read_csv(r"D:\Project4\kidney_disease.csv")
#kdata.head(5)

# Remove feature id as there is no impact
kdata = kdata.drop('id', axis=1)

# New data
new_kdata = kdata

# Label Encoding
le = LabelEncoder()
new_kdata['rbc'] = le.fit_transform(new_kdata['rbc'])
new_kdata['pc'] = le.fit_transform(new_kdata['pc'])
new_kdata['pcc'] = le.fit_transform(new_kdata['pcc'])
new_kdata['ba'] = le.fit_transform(new_kdata['ba'])
new_kdata['htn'] = le.fit_transform(new_kdata['htn'])
new_kdata['dm'] = le.fit_transform(new_kdata['dm'])
new_kdata['cad'] = le.fit_transform(new_kdata['cad'])
new_kdata['appet'] = le.fit_transform(new_kdata['appet'])
new_kdata['pe'] = le.fit_transform(new_kdata['pe'])
new_kdata['ane'] = le.fit_transform(new_kdata['ane'])
new_kdata['classification'] = le.fit_transform(new_kdata['classification'])
#new_kdata.head(5)

# Converting Object to numeric
new_kdata['pcv'] = pd.to_numeric(new_kdata['pcv'], errors='coerce')
new_kdata['wc'] = pd.to_numeric(new_kdata['wc'], errors='coerce')
new_kdata['rc'] = pd.to_numeric(new_kdata['rc'], errors='coerce')

# Handling Null data
for col in new_kdata.columns:
    new_kdata[col] = new_kdata[col].fillna(new_kdata[col].median())
#new_kdata.head(5)

# To check Outliers
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(12,10))
axes = axes.flatten()

for ax, col in zip(axes,new_kdata.columns):
    sns.boxplot(new_kdata[col], ax=ax)

for ax in axes[len(new_kdata.columns):]:
    ax.set_visible(False)

plt.show()
plt.tight_layout()

# To get Outlier columns
outlier_cols = []

for col in new_kdata.columns:
    Q1 = new_kdata[col].quantile(0.25)
    Q3 = new_kdata[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # check if any values are outside bounds
    if ((new_kdata[col] < lower) | (new_kdata[col] > upper)).any():
        outlier_cols.append(col)

#outlier_cols

# Handling Outliers with log trasformation
for col in outlier_cols:
    # Shift column if needed
    if (new_kdata[col] <= 0).any():
        new_kdata[col] = new_kdata[col] - new_kdata[col].min() + 1
        
    new_kdata[col] = np.log(new_kdata[col])
   
#new_kdata

# Features and Target
A = new_kdata.drop(['classification'], axis=1)
b = new_kdata['classification']

# SMOTE Technique
smote = SMOTETomek()
A_smote, b_smote = smote.fit_resample(A,b)

# Split Test and Train
A_train,A_test,b_train,b_test = train_test_split(A_smote,b_smote, test_size=0.3, random_state=99)

# Pickle file loading
@st.cache_resource
def load_kidney_model():
    with open("D:/Project4/kidney_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

kidney_model = load_kidney_model()
print(type(kidney_model))

# **********************************PARKINSONS DISEASE PREDICTION*************************

# Fetch the data
pkdata = pd.read_csv(r"D:\Project4\parkinsons.csv")
pkdata.head(5)

# Remove feature name as there is no impact
pkdata = pkdata.drop('name', axis=1)

# To check null values
pkdata.isnull().sum()

# New data
new_pkdata = pkdata

# To check Outliers
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(12,10))
axes = axes.flatten()

for ax, col in zip(axes,new_pkdata.columns):
    if col != 'status':
        sns.boxplot(new_pkdata[col], ax=ax)

for ax in axes[len(new_pkdata.columns):]:
    ax.set_visible(False)

plt.show()
plt.tight_layout()

# To get Outlier columns
outlier_cols = []

for col in new_pkdata.columns:
    if col == 'status':
       continue
    if not pd.api.types.is_numeric_dtype(new_pkdata[col]):
         continue
    Q1 = new_pkdata[col].quantile(0.25)
    Q3 = new_pkdata[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
# check if any values are outside bounds
    if ((new_pkdata[col] < lower) | (new_pkdata[col] > upper)).any():
        outlier_cols.append(col)

#outlier_cols

# Handling Outliers with log trasformation
for col in outlier_cols:
    # Shift column if needed
    if (new_pkdata[col] <= 0).any():
        new_pkdata[col] = new_pkdata[col] - new_pkdata[col].min() + 1
        
    new_pkdata[col] = np.log(new_pkdata[col])

# Features and Target
F = new_pkdata.drop(['status'], axis=1)
t = new_pkdata['status']

# SMOTE Technique
smote = SMOTETomek()
F_smote, t_smote = smote.fit_resample(F,t)

# Split Test and Train
F_train,F_test,t_train,t_test = train_test_split(F_smote,t_smote, test_size=0.3, random_state=42)

# Pickle File Loading
@st.cache_resource
def load_parkinsons_model():
    with open("D:/Project4/parkinsons_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

parkinsons_model = load_parkinsons_model()
print(type(parkinsons_model))


# **********************************STREAMLIT UI******************************************

# Streamlit
side = st.sidebar.radio(
    "Navigation",
    ["HOME","LIVER DISEASE PREDICTION", "KIDNEY DISEASE PREDICTION", "PARKINSONS DISEASE PREDICTION"]
)

# Back ground image
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# On choosing sidebar navigation
if side == "HOME":
    set_bg("D:/Project4/image2.jpg")
    st.markdown("""
    <h1 style="text-align: center; color: white">MULTIPLE DISEASE PREDICTION</h1>
    <h2 style="text-align: center; color: white;"> WELCOME!</h2>   
    <h3 style="text-align: center; color: white;">This application is designed to predict multiple diseases 
                such as Liver Disease, Kidney Disease, and Parkinson’s Disease using machine learning techniques. 
                By inputting relevant medical test parameters, the system analyzes the data and predicts the likelihood 
                of a patient having a specific disease. The goal of the application is to assist in early disease detection 
                and support healthcare decision-making. </h3> 
    
    </style>               
    """,unsafe_allow_html=True)
if side == "LIVER DISEASE PREDICTION":
    set_bg("D:/Project4/image3.jpg")
    st.markdown("""
    <h1 style="text-align: center; color: white">LIVER DISEASE PREDICTION</h1>   
    </style>               
    """,unsafe_allow_html=True)

    st.markdown("<h3 style='color:white;'>Enter Patient Details</h3>", unsafe_allow_html=True)

    # Input fields
    Age = st.number_input("Age", min_value=1, max_value=100)
    Gender = st.selectbox("Gender", ["Female", "Male"])
    Total_Bilirubin = st.number_input("Total Bilirubin")
    Direct_Bilirubin = st.number_input("Direct Bilirubin")
    Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase")
    Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase")
    Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase")
    Total_Protiens = st.number_input("Total Proteins")
    Albumin = st.number_input("Albumin")
    Albumin_and_Globulin_Ratio = st.number_input("Albumin and Globulin Ratio")

    # Encode gender
    Gender = 1 if Gender == "Male" else 0

    # Predict button
    if st.button("Predict Liver Disease"):
        input_data = np.array([[Age, Gender, Total_Bilirubin, Direct_Bilirubin,
                                 Alkaline_Phosphotase, Alamine_Aminotransferase,
                                 Aspartate_Aminotransferase, Total_Protiens,
                                 Albumin, Albumin_and_Globulin_Ratio]])

        prediction = liver_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ The patient is likely to have Liver Disease")
        else:
            st.success("✅ The patient is not likely to have Liver Disease")

if side == "KIDNEY DISEASE PREDICTION":
    set_bg("D:/Project4/image4.jpg")
    st.markdown("""
    <h1 style="text-align: center; color: white">KIDNEY DISEASE PREDICTION</h1>   
    </style>               
    """,unsafe_allow_html=True)

    st.markdown("<h3 style='color:white;'>Enter Patient Details</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 100)
        bp = st.number_input("Blood Pressure", 50, 200)
        sg = st.selectbox("Specific Gravity", [1.005,1.010,1.015,1.020,1.025])
        al = st.selectbox("Albumin", [0,1,2,3,4,5])
        su = st.selectbox("Sugar", [0,1,2,3,4,5])

    with col2:
        rbc = st.selectbox("Red Blood Cells", ['normal','abnormal'])
        pc = st.selectbox("Pus Cell", ['normal','abnormal'])
        pcc = st.selectbox("Pus Cell Clumps", ['present','notpresent'])
        ba = st.selectbox("Bacteria", ['present','notpresent'])
        bgr = st.number_input("Blood Glucose Random", 50, 500)

    with col3:
        bu = st.number_input("Blood Urea", 1, 400)
        sc = st.number_input("Serum Creatinine", 0.1, 20.0)
        sod = st.number_input("Sodium", 100, 200)
        pot = st.number_input("Potassium", 2.0, 10.0)
        hemo = st.number_input("Hemoglobin", 3.0, 20.0)

    pcv = st.number_input("Packed Cell Volume", 10, 60)
    wc = st.number_input("White Blood Cell Count", 3000, 20000)
    rc = st.number_input("Red Blood Cell Count", 2.0, 8.0)

    htn = st.selectbox("Hypertension", ['yes','no'])
    dm = st.selectbox("Diabetes Mellitus", ['yes','no'])
    cad = st.selectbox("Coronary Artery Disease", ['yes','no'])
    appet = st.selectbox("Appetite", ['good','poor'])
    pe = st.selectbox("Pedal Edema", ['yes','no'])
    ane = st.selectbox("Anemia", ['yes','no'])

    rbc_map = {'normal': 1, 'abnormal': 0}
    pc_map = {'normal': 1, 'abnormal': 0}
    pcc_map = {'present': 1, 'notpresent': 0}
    ba_map = {'present': 1, 'notpresent': 0}
    htn_map = {'yes': 1, 'no': 0}
    dm_map = {'yes': 1, 'no': 0}
    cad_map = {'yes': 1, 'no': 0}
    appet_map = {'good': 1, 'poor': 0}
    pe_map = {'yes': 1, 'no': 0}
    ane_map = {'yes': 1, 'no': 0}

    if st.button("Predict Kidney Disease"):
        input_data = pd.DataFrame([{
            'age': age,
            'bp': bp,
            'sg': sg,
            'al': al,
            'su': su,
            'rbc': rbc_map[rbc],
            'pc': pc_map[pc],
            'pcc': pcc_map[pcc],
            'ba': ba_map[ba],
            'bgr': bgr,
            'bu': bu,
            'sc': sc,
            'sod': sod,
            'pot': pot,
            'hemo': hemo,
            'pcv': pcv,
            'wc': wc,
            'rc': rc,
            'htn': htn_map[htn],
            'dm': dm_map[dm],
            'cad': cad_map[cad],
            'appet': appet_map[appet],
            'pe': pe_map[pe],
            'ane': ane_map[ane]
        }])

        prediction = kidney_model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ Kidney Disease Detected")
        else:
            st.success("✅ No Kidney Disease Detected")
if side == "PARKINSONS DISEASE PREDICTION":
    set_bg("D:/Project4/image5.jpg")
    st.markdown("""
    <h1 style="text-align: center">PARKINSONS DISEASE PREDICTION</h1>   
    </style>               
    """,unsafe_allow_html=True)

    st.markdown("<h3>Enter Patient Details</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    # -------- COLUMN 1 --------
    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", value=float(new_pkdata['MDVP:Fo(Hz)'].median()))
        fhi = st.number_input("MDVP:Fhi(Hz)", value=float(new_pkdata['MDVP:Fhi(Hz)'].median()))
        flo = st.number_input("MDVP:Flo(Hz)", value=float(new_pkdata['MDVP:Flo(Hz)'].median()))
        jitter = st.number_input("MDVP:Jitter(%)", value=float(new_pkdata['MDVP:Jitter(%)'].median()))
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=float(new_pkdata['MDVP:Jitter(Abs)'].median()))
        rap = st.number_input("MDVP:RAP", value=float(new_pkdata['MDVP:RAP'].median()))
        ppq = st.number_input("MDVP:PPQ", value=float(new_pkdata['MDVP:PPQ'].median()))

    # -------- COLUMN 2 --------
    with col2:
        ddp = st.number_input("Jitter:DDP", value=float(new_pkdata['Jitter:DDP'].median()))
        shimmer = st.number_input("MDVP:Shimmer", value=float(new_pkdata['MDVP:Shimmer'].median()))
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=float(new_pkdata['MDVP:Shimmer(dB)'].median()))
        apq3 = st.number_input("Shimmer:APQ3", value=float(new_pkdata['Shimmer:APQ3'].median()))
        apq5 = st.number_input("Shimmer:APQ5", value=float(new_pkdata['Shimmer:APQ5'].median()))
        apq = st.number_input("MDVP:APQ", value=float(new_pkdata['MDVP:APQ'].median()))
        dda = st.number_input("Shimmer:DDA", value=float(new_pkdata['Shimmer:DDA'].median()))

    # -------- COLUMN 3 --------
    with col3:
        nhr = st.number_input("NHR", value=float(new_pkdata['NHR'].median()))
        hnr = st.number_input("HNR", value=float(new_pkdata['HNR'].median()))
        rpde = st.number_input("RPDE", value=float(new_pkdata['RPDE'].median()))
        dfa = st.number_input("DFA", value=float(new_pkdata['DFA'].median()))
        spread1 = st.number_input("spread1", value=float(new_pkdata['spread1'].median()))
        spread2 = st.number_input("spread2", value=float(new_pkdata['spread2'].median()))
        d2 = st.number_input("D2", value=float(new_pkdata['D2'].median()))
    ppe = st.number_input("PPE", value=float(new_pkdata['PPE'].median()))

    # -------- PREDICTION --------
    if st.button("Predict Parkinson's Disease"):

        input_df = pd.DataFrame([{
            'MDVP:Fo(Hz)': fo,
            'MDVP:Fhi(Hz)': fhi,
            'MDVP:Flo(Hz)': flo,
            'MDVP:Jitter(%)': jitter,
            'MDVP:Jitter(Abs)': jitter_abs,
            'MDVP:RAP': rap,
            'MDVP:PPQ': ppq,
            'Jitter:DDP': ddp,
            'MDVP:Shimmer': shimmer,
            'MDVP:Shimmer(dB)': shimmer_db,
            'Shimmer:APQ3': apq3,
            'Shimmer:APQ5': apq5,
            'MDVP:APQ': apq,
            'Shimmer:DDA': dda,
            'NHR': nhr,
            'HNR': hnr,
            'RPDE': rpde,
            'DFA': dfa,
            'spread1': spread1,
            'spread2': spread2,
            'D2': d2,
            'PPE': ppe
        }])

        # ---- APPLY SAME LOG TRANSFORM USED IN TRAINING ----
        
        prediction = parkinsons_model.predict(input_df)[0]

        if prediction == 1:
            st.error("⚠️ Parkinson's Disease Detected")
        else:
            st.success("✅ No Parkinson's Disease Detected")