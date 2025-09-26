import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Loading the insurance dataset
insurance_data_path = 'insurance.csv'
ins = pd.read_csv(insurance_data_path)
validation_df = pd.read_csv('validation_dataset.csv')

ins['charges'] = pd.to_numeric(ins['charges'].astype(str).str.replace('$', '', regex=False), errors='coerce')
ins['sex'] = ins['sex'].replace(
    {'M':'male', 'man':'male',
    'F':'female', 'woman':'female'}
)
ins['region'] = ins['region'].str.lower()
ins = ins[ins['age']>=0 & (ins['children']>=0)]
ins.dropna(inplace=True)

features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
target = 'charges'

X = ins[features]
y = ins[target]

X_encoded = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)

model = LinearRegression()
model.fit(X_encoded, y)

y_pred_train = model.predict(X_encoded)
r2_score = r2_score(y, y_pred_train)

print(f"R-Squared Score on training data: {r2_score:.4f}")

validation_data = validation_df.copy()
validation_X_encoded = pd.get_dummies(validation_df, columns=['sex', 'smoker', 'region'], drop_first=True)

missing_cols = set(X_encoded.columns) - set(validation_X_encoded.columns)
for c in missing_cols:
    validation_X_encoded[c] = 0
validation_X_encoded = validation_X_encoded[X_encoded.columns]

predicted_charges = model.predict(validation_X_encoded)

predicted_charges = np.maximum(1000, predicted_charges)

validation_data['predicted_charges'] = predicted_charges

print("\nPredictions for the validation dataset:")
print(validation_data.head())