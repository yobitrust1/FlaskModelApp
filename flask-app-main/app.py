# import libraries
from flask import Flask,request,session,redirect,url_for
from flask import jsonify
import time
import pickle
import json 
import sklearn
from sklearn import preprocessing
from xgboost import XGBClassifier
from flask_cors import CORS
import pandas as pd
#pip install scikit-learn==0.24.1
#pip install xgboost==0.90
#pip install numpy==1.19.5
#pip install pandas==1.1.5
app = Flask(__name__)
cors = CORS(app, resources={r"/predict": {"origins": "*"}})

global prediction
model = pickle.load(open('best_XGBClassifier_target.pickle', 'rb'))
scaler = pickle.load(open('min_max_scaler scaling.pkl', 'rb'))
imputer = pickle.load(open('imputer.pickle', 'rb'))
yan_model = pickle.load(open('yan_XGBClassifier.pickle', 'rb'))
@app.route("/predict",methods=["POST"])
def index():
    if request.method == "POST":
        request_data= json.loads(request.data.decode('utf-8'))
        X_testt= list(request_data.values())
        print(X_testt)
        X_testt=pd.DataFrame(X_testt)
        X_testt=X_testt.T    
        X_test_scaled=scaler.transform(X_testt)
        X_test_scaled=pd.DataFrame(X_test_scaled)
        X_test_scaled.columns=['Laboratory_test_L', 'Laboratory_test_N', 'Laboratory_test_ESR_(mm/hr)', 'Laboratory_test_CRP_(mg/L)', 'Laboratory_test_PCT_(ng/ml)', 'Laboratory_test_CK_MB_(ng/ml)', 'Laboratory_test_D_dimer_(ug/ml)', 'Laboratory_test_ALT_(U/L)', 'Laboratory_test_AST_(U/L)', 'Laboratory_test_ALB_(g/L)', 'Laboratory_test_LDH_(U/L)', 'Age', 'CK', 'O2%', 'Symptoms_Cough', 'Symptoms_Dyspnea', 'Symptoms_Fatigue'] 
        pred = model.predict(X_test_scaled)
        if pred ==0:
            prediction="not_severe"
        if pred ==1 :
            prediction="severe"
   
        yan_features=['Laboratory_test_CRP_(mg/L)','Laboratory_test_LDH_(U/L)','Laboratory_test_L']
        yan_X_test=X_test_scaled[yan_features]
   
        pred2 = yan_model.predict(yan_X_test)
        if pred2 ==0:
            prediction2="not_severe"
        if pred2 ==1 :
            prediction2="severe"
        
        result=jsonify({'pred':prediction ,'pred2':prediction2})
        print('***********',prediction )



        return result


 
 
 
 
 