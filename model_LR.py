import numpy as np 
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


#load data

df = pd.read_csv(r"/Users/akshatthakur22/Desktop/python/project_1/dataset/Housing.csv")


#check missing data
print("Missing data values -> ")
print(df.isnull().sum())


#handle missing valuses 
df['price'] = df['price'].fillna(df['price'].mean())
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].mode()[0])
df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].mode()[0])
df['stories'] = df['stories'].fillna(df['stories'].mode()[0])


#recheck missing data 
print("Missing data values -> ")
print(df.isnull().sum())



##check data type is shoule be only in intger or numrical 

print(df.info())

## encoding variable / object / string in form of numbers intger....
# 0s or 1s 

not_int_col = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]

LE = LabelEncoder()

for col in not_int_col:
    df[col] = LE.fit_transform(df[col])


#recheck for confirmation
print(df.info())


#define x and y 
x = df.drop("price", axis=1)   # input features
y = df["price"]                # target variable


#  Split Dataset (Train / Test).  (80)/(20)

x_train, x_test, y_train, y_test = train_test_split( x , y , test_size= .2 , random_state=32 )


##create pipline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
]) 


#train model 
pipeline.fit(x_train,y_train)


##predection of model
y_predection = pipeline.predict(x_test)


# secure your model :
with open("house_price_pipeline.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("\nPipeline model saved successfully!")
