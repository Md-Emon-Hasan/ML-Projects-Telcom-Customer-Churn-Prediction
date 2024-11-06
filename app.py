from flask import Flask
from flask import request
from flask import render_template
import pickle
import numpy as np

df = pickle.load(open('./models/df.pkl','rb'))
classifier = pickle.load(open('./models/classifier.pkl','rb'))

app = Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    # Sort the unique values
    SeniorCitizen = sorted(df['SeniorCitizen'].unique())
    Dependents = sorted(df['Dependents'].unique())
    Contract_MonthToMonth = sorted(df['Contract_MonthToMonth'].unique())
    Contract_One_year = sorted(df['Contract_One_year'].unique())
    Contract_Two_year = sorted(df['Contract_Two_year'].unique())
    tenure_bin_New = sorted(df['tenure_bin_New'].unique())
    tenure_bin_Mid = sorted(df['tenure_bin_Mid'].unique())
    tenure_bin_Long = sorted(df['tenure_bin_Long'].unique())
    InternetService_Fiber_optic = sorted(df['InternetService_Fiber_optic'].unique())
    OnlineSecurity = sorted(df['OnlineSecurity'].unique())
    TechSupport = sorted(df['TechSupport'].unique())
    PaymentMethod_Electronic_check = sorted(df['PaymentMethod_Electronic_check'].unique())
    PaperlessBilling = sorted(df['PaperlessBilling'].unique())
    
    return render_template('index.html', SeniorCitizen=SeniorCitizen, Dependents=Dependents, Contract_MonthToMonth=Contract_MonthToMonth, Contract_One_year=Contract_One_year, Contract_Two_year=Contract_Two_year, tenure_bin_New=tenure_bin_New, tenure_bin_Mid=tenure_bin_Mid, tenure_bin_Long=tenure_bin_Long, InternetService_Fiber_optic=InternetService_Fiber_optic, OnlineSecurity=OnlineSecurity, TechSupport=TechSupport, PaymentMethod_Electronic_check=PaymentMethod_Electronic_check, PaperlessBilling=PaperlessBilling)

@app.route('/predict',methods=['POST'])
def predict():
    SeniorCitizen = int(request.form['SeniorCitizen'])
    Dependents = int(request.form['Dependents'])
    Contract_MonthToMonth = float(request.form['Contract_MonthToMonth'])
    Contract_One_year = float(request.form['Contract_One_year'])
    Contract_Two_year = float(request.form['Contract_Two_year'])
    tenure_bin_New = float(request.form['tenure_bin_New'])
    tenure_bin_Mid = float(request.form['tenure_bin_Mid'])
    tenure_bin_Long = float(request.form['tenure_bin_Long'])
    InternetService_Fiber_optic = float(request.form['InternetService_Fiber_optic'])
    OnlineSecurity = int(request.form['OnlineSecurity'])
    TechSupport = int(request.form['TechSupport'])
    PaymentMethod_Electronic_check = float(request.form['PaymentMethod_Electronic_check'])
    PaperlessBilling = int(request.form['PaperlessBilling'])
    MonthlyCharges = float(request.form['MonthlyCharges'])
    TotalCharges = float(request.form['TotalCharges'])
    
    # predict
    predict_churn = classifier.predict(np.array([SeniorCitizen, Dependents, Contract_One_year, Contract_MonthToMonth, Contract_Two_year, tenure_bin_New, tenure_bin_Mid, tenure_bin_Long, InternetService_Fiber_optic, OnlineSecurity, TechSupport, PaymentMethod_Electronic_check, PaperlessBilling, MonthlyCharges, TotalCharges]).reshape(1,15))
    
    return render_template(
        'index.html',
        predict_churn=predict_churn,
        SeniorCitizen=SeniorCitizen,
        Dependents=Dependents,
        Contract_MonthToMonth=Contract_MonthToMonth,
        Contract_One_year=Contract_One_year,
        Contract_Two_year=Contract_Two_year,
        tenure_bin_New=tenure_bin_New,
        tenure_bin_Mid=tenure_bin_Mid,
        tenure_bin_Long=tenure_bin_Long,
        InternetService_Fiber_optic=InternetService_Fiber_optic,
        OnlineSecurity=OnlineSecurity,
        TechSupport=TechSupport,
        PaymentMethod_Electronic_check=PaymentMethod_Electronic_check,
        PaperlessBilling=PaperlessBilling,
        MonthlyCharges=MonthlyCharges,
        TotalCharges=TotalCharges
        )

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)