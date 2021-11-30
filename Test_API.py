import uvicorn

from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas
import eeweather
import json


class Inputs(BaseModel):
    value1: float
    value2: float
    x: str
    y: str
    jsonCSV: str

app = FastAPI()

origins = [
  "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def add(num1,num2):
    return num1+num2

@app.get('/')

def index():

    return {'message': "This is the home page of this API."}

@app.get('/calculate')

def process(value1: float, value2: float):

    Sum = add(value1,value2)

    return {'result': Sum}

@app.post('/sumCSV')

def sum_CSV(file: UploadFile = File(...)):
    df = pandas.read_csv(file.filename,index_col=0, header=0)
    Sum = df.sum()


    return {'result': float(Sum)}

@app.post('/sumJSON')

def sum_JSON(jsonCSV: str):
    JSONData = json.loads(jsonCSV)
    df = pandas.DataFrame.from_dict(JSONData['data'])
    df["Index"] = pandas.to_numeric(df["Index"], downcast="float")
    df["Value"] = pandas.to_numeric(df["Value"], downcast="float")

    Sum = df['Value'].sum()


    return {'result': float(Sum)}


@app.get('/linearregression')

def regress(x: str, y: str):
    x_np = np.fromstring(x, dtype=float, sep=',')
    y_np = np.fromstring(y, dtype=float, sep=',')
    # x_np = np.array(x)
    # y_np = np.array(y)

    x_reg = np.column_stack((np.ones([np.size(x_np), 1]), x_np.transpose()))
    beta = np.linalg.solve(np.matmul(x_reg.transpose(), x_reg), np.matmul(x_reg.transpose(), y_np.transpose()))

    return {'result': str(beta)}

@app.get('/apiv2/')

@app.get("/weather")
def getWeatherStation(zip: str):
    try:
        lat, long = eeweather.zcta_to_lat_long(zip)
        ranked_stations = eeweather.rank_stations(lat, long)
        ranked_stations.drop(ranked_stations[ranked_stations['is_tmy3'] == False].index, inplace=True) # removes all non-TMY3

        Nearest_TMY3_Station = ranked_stations.iloc[0]
        Weather_Output = Nearest_TMY3_Station

    except Exception as e:
        Weather_Output = "Invalid location."
    return str(Weather_Output)

if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)