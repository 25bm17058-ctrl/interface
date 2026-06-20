from tensorflow.keras.models import load_model
model = load_model("model.h5")

import joblib
scaler = joblib.load("x_scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")


Gender = input("Gender: ").strip()
Age = int(input("Age: "))
Height = float(input("Height: "))
Weight = float(input("Weight: "))
CALC = input("Alcohol consumption: ").strip()
FAVC = input("High calorie food (yes/no): ").strip()
FCVC = float(input("Vegetable consumption (1-3): "))
NCP = float(input("Meals per day: "))
SCC = input("Calories monitoring (yes/no): ").strip()
SMOKE = input("Smoke (yes/no): ").strip()
CH2O = float(input("Water intake: "))
family_history = input("Family history (yes/no): ").strip()
FAF = float(input("Physical activity: "))
TUE = float(input("Technology usage: "))
CAEC = input("Food between meals: ").strip()
MTRANS = input("Transport: ").strip()

Gender = label_encoders['Gender'].transform([Gender])[0]
CALC = label_encoders['CALC'].transform([CALC])[0]
FAVC = label_encoders['FAVC'].transform([FAVC])[0]
SCC = label_encoders['SCC'].transform([SCC])[0]
SMOKE = label_encoders['SMOKE'].transform([SMOKE])[0]
family_history = label_encoders['family_history_with_overweight'].transform([family_history])[0]
CAEC = label_encoders['CAEC'].transform([CAEC])[0]
MTRANS = label_encoders['MTRANS'].transform([MTRANS])[0]

import pandas as pd
new_data = pd.DataFrame([[
    Gender, Age, Height, Weight,
    CALC, FAVC, FCVC, NCP,
    SCC, SMOKE, CH2O, family_history,
    FAF, TUE, CAEC, MTRANS
]])
new_data_scaled = scaler.transform(new_data)


prediction = model.predict(new_data_scaled)[0]

highest = max(prediction)

if prediction[0] == highest:
    result = 0
elif prediction[1] == highest:
    result = 1
elif prediction[2] == highest:
    result = 2
elif prediction[3] == highest:
    result = 3
elif prediction[4] == highest:
    result = 4
elif prediction[5] == highest:
    result = 5
else:
    result = 6


output = label_encoders['NObeyesdad'].inverse_transform([result])[0]
print("Predicted Class:", output)