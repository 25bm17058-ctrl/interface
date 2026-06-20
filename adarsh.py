import pandas as pd
data=pd.read_csv("ObesityDataSet_raw_and_data_sinthetic (1).csv")
print(data.head())
print(f"Number of duplicate rows: {data.duplicated().sum()}")
data = data.drop_duplicates()
print(data.isnull().mean()*100)
print(f"Number of duplicate rows: {data.duplicated().sum()}")

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])
label_encoders['Gender'] = le_gender
le_calc = LabelEncoder()
data['CALC'] = le_calc.fit_transform(data['CALC'])
label_encoders['CALC'] = le_calc
le_favc = LabelEncoder()
data['FAVC'] = le_favc.fit_transform(data['FAVC'])
label_encoders['FAVC'] = le_favc
le_scc = LabelEncoder()
data['SCC'] = le_scc.fit_transform(data['SCC'])
label_encoders['SCC'] = le_scc
le_smoke = LabelEncoder()
data['SMOKE'] = le_smoke.fit_transform(data['SMOKE'])
label_encoders['SMOKE'] = le_smoke
le_family = LabelEncoder()
data['family_history_with_overweight'] = le_family.fit_transform(data['family_history_with_overweight'])
label_encoders['family_history_with_overweight'] = le_family
le_caec = LabelEncoder()
data['CAEC'] = le_caec.fit_transform(data['CAEC'])
label_encoders['CAEC'] = le_caec
le_mtrans = LabelEncoder()
data['MTRANS'] = le_mtrans.fit_transform(data['MTRANS'])
label_encoders['MTRANS'] = le_mtrans
le_target = LabelEncoder()
data['NObeyesdad'] = le_target.fit_transform(data['NObeyesdad'])
label_encoders['NObeyesdad'] = le_target
print(data.head())
x = data.drop(columns=["NObeyesdad"])
y = data['NObeyesdad']
print(x.head())
print(y.head())

from sklearn.preprocessing import StandardScaler
x_scaler=StandardScaler()
x_scaled=x_scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential ([
    Dense(16,  input_shape=(16,)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(7, activation="softmax")
])
model.compile(
    loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(x_train, y_train, epochs=200)
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

import joblib

joblib.dump(x_scaler, 'x_scaler.pkl')
joblib.dump(label_encoders, "label_encoders.pkl")

import tensorflow as tf
model.save('model.h5')