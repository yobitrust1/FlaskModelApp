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
cors = CORS(app, resources={r"/predict": {"origins":"http://localhost:3000"}})
global prediction
model = pickle.load(open('predict mortality_f_0.68.pickle', 'rb'))
scaler = pickle.load(open('full_pipeline11_feat.pickle', 'rb'))
@app.route("/predict",methods=["POST"])
def index():
    if request.method == "POST":
        request_data= json.loads(request.data.decode('utf-8'))
        X_testt= list(request_data.values())
        X_testt=pd.DataFrame(X_testt)
        X_testt=X_testt.T    
        print(X_testt)
        X_test_scaled=scaler.transform(X_testt)
        X_test_scaled=pd.DataFrame(X_test_scaled)
        X_test_scaled.columns=['Mg2_1','LDH1','GB1','BiliT1','créat1','ASAT1','fibrinogène1','pH1','température']
        pred = model.predict(X_test_scaled)
        if pred ==0:
            prediction="survived"
        if pred ==1 :
            prediction="not survived"
        print("mortality prediction",prediction)
        result=jsonify({'pred':prediction})
        print("********",prediction)
        return result 


 
 
 
