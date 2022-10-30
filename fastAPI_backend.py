from pydantic import BaseModel, validator
from fastapi import FastAPI
import pandas as pd
import numpy as np
import uvicorn
import pickle
from fastapi.responses import JSONResponse
from fastapi.responses import ORJSONResponse
from typing import Optional


with open('model_pipeline.pickle', 'rb') as handle:
    pipeline = pickle.load(handle)

#df_data = pd.read_csv('df_application_test.zip', compression='zip', header=0, sep=',', quotechar='"')

def replace_none(test_dict):
    # checking for dictionary and replacing if None
    if isinstance(test_dict, dict):
        for key in test_dict:
            if test_dict[key] is None:
                test_dict[key] = np.nan
    return test_dict
            
class Frontend_data(BaseModel):
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: Optional[float]
    CODE_GENDER: Optional[bool]
    DAYS_BIRTH: Optional[int]
    NAME_FAMILY_STATUS: Optional[str]
    NAME_EDUCATION_TYPE: Optional[str]
    ORGANIZATION_TYPE: Optional[str]
    DAYS_EMPLOYED: Optional[int]
    ACTIVE_AMT_CREDIT_SUM_DEBT_MAX: Optional[float]
    PAYMENT_RATE: Optional[float]
    ANNUITY_INCOME_PERC: Optional[float]
    DAYS_ID_PUBLISH: Optional[int]
    REGION_POPULATION_RELATIVE: Optional[float]
    FLAG_OWN_CAR: Optional[bool]
    OWN_CAR_AGE: Optional[float]
    FLAG_DOCUMENT_3: Optional[bool]
    CLOSED_DAYS_CREDIT_MAX: Optional[float]
    INSTAL_AMT_PAYMENT_SUM: Optional[float]
    APPROVED_CNT_PAYMENT_MEAN: Optional[float]
    PREV_CNT_PAYMENT_MEAN: Optional[float]
    PREV_APP_CREDIT_PERC_MIN: Optional[float]
    INSTAL_DPD_MEAN: Optional[float]
    INSTAL_DAYS_ENTRY_PAYMENT_MAX: Optional[float]
    POS_MONTHS_BALANCE_SIZE: Optional[float]

class ID(BaseModel):
    SK_ID_CURR: int

#app = FastAPI(default_response_class=ORJSONResponse)
app = FastAPI()

@app.get('/')
def index():
    return {'message': "You've entered the API backend"}
'''
@app.get('/get_ids')
async def get_ids():
    return {'IDs': df_data["SK_ID_CURR"].tolist()}

@app.post('/get_data', response_class=ORJSONResponse)
async def get_data(data: ID):
    return {'Data': [df_data.loc[df_data["SK_ID_CURR"] == data.dict()["SK_ID_CURR"]].to_dict(orient='records')]}
'''
@app.post('/predict')
async def predict_risk(data: Frontend_data):
    dict_data = replace_none(data.dict())
    df = pd.DataFrame([dict_data])
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
    pred = pipeline.predict_proba(df)[:, 1].tolist()
    result = {"Probability": pred}
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    