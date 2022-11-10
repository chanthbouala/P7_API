from pydantic import BaseModel, validator
from fastapi import FastAPI
import pandas as pd
import numpy as np
import uvicorn
import pickle
from fastapi.responses import JSONResponse
from fastapi.responses import ORJSONResponse
from typing import Optional
from sklearn.preprocessing import StandardScaler
from functions import find_knn

with open('model_pipeline.pickle', 'rb') as handle:
    pipeline = pickle.load(handle)

df_data_selection = pd.read_csv('df_application_test_2_selection.zip', compression='zip', header=0, sep=',', quotechar='"')
for col in df_data_selection.columns:
    if col != "SK_ID_CURR":
        if df_data_selection[col].dtype == 'object' or df_data_selection[col].dtype == 'str':
            df_data_selection[col] = df_data_selection[col].astype('category')
        elif col == "FLAG_DOCUMENT_3" or col == "FLAG_OWN_CAR" or col == "CODE_GENDER":
            df_data_selection[col] = df_data_selection[col].astype(bool)
        else:
            df_data_selection[col] = df_data_selection[col].astype(np.float32)
    else:
        df_data_selection[col] = df_data_selection[col].astype(np.int32)

df_data_knn_ohe = pd.read_csv('df_application_test_knn_ohe.zip', compression='zip', header=0, sep=',', quotechar='"')  
ss = StandardScaler()
data_knn_ohe_scaled = ss.fit_transform(df_data_knn_ohe.drop("SK_ID_CURR", axis=1))
df_data_knn_ohe_scaled = pd.DataFrame(data_knn_ohe_scaled, columns=df_data_knn_ohe.drop("SK_ID_CURR", axis=1).columns)
df_data_knn_ohe_scaled = pd.concat([df_data_knn_ohe["SK_ID_CURR"], df_data_knn_ohe_scaled], axis=1)

def replace_none(test_dict):
    # checking for dictionary and replacing if None
    if isinstance(test_dict, dict):
        for key in test_dict:
            if test_dict[key] is None:
                test_dict[key] = np.nan
    return test_dict
            
class Frontend_data(BaseModel):
    ACTIVE_AMT_CREDIT_SUM_DEBT_MAX: Optional[float]
    AMT_ANNUITY: Optional[float]
    AMT_CREDIT: Optional[float]
    AMT_GOODS_PRICE: Optional[float]
    ANNUITY_INCOME_PERC: Optional[float]
    APPROVED_CNT_PAYMENT_MEAN: Optional[float]
    CLOSED_DAYS_CREDIT_MAX: Optional[float]
    CODE_GENDER: Optional[bool]
    DAYS_BIRTH: Optional[int]
    DAYS_EMPLOYED: Optional[int]
    DAYS_ID_PUBLISH: Optional[int]
    FLAG_DOCUMENT_3: Optional[bool]
    FLAG_OWN_CAR: Optional[bool]
    INSTAL_AMT_PAYMENT_SUM: Optional[float]
    INSTAL_DAYS_ENTRY_PAYMENT_MAX: Optional[float]
    INSTAL_DPD_MEAN: Optional[float]
    NAME_EDUCATION_TYPE: Optional[str]
    NAME_FAMILY_STATUS: Optional[str]
    ORGANIZATION_TYPE: Optional[str]
    OWN_CAR_AGE: Optional[float]
    PAYMENT_RATE: Optional[float]
    POS_MONTHS_BALANCE_SIZE: Optional[float]
    PREV_APP_CREDIT_PERC_MIN: Optional[float]
    PREV_CNT_PAYMENT_MEAN: Optional[float]
    REGION_POPULATION_RELATIVE: Optional[float]
    AMT_INCOME_TOTAL: Optional[float]

class ID(BaseModel):
    SK_ID_CURR: int

app = FastAPI()

@app.get('/')
def index():
    return {'message': "You've entered the API backend"}

@app.post('/predict')
async def predict_risk(data: Frontend_data):
    dict_data = replace_none(data.dict())
    df = pd.DataFrame([dict_data])
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'str':
            df[col] = df[col].astype('category')
    pred = pipeline.predict_proba(df)[:, 1].tolist()
    result = {"Probability": pred}
    return result

@app.post('/find_knn')
async def calculate_knn(data: ID):
    dict_data = data.dict()
    knns = find_knn(dict_data["SK_ID_CURR"], df_data_knn_ohe_scaled, k=21)
    
    y_pred_all = pipeline.predict_proba(df_data_selection.drop("SK_ID_CURR", axis=1))[:, 1].tolist()
    results = {"knns": knns,
              "y_pred_all": y_pred_all}
    
    return results

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)    