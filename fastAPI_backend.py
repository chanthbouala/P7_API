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

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    