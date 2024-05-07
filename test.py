# %%
from igann import IGANN
from igann import GAMmodel

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import pandas as pd

from pprint import pprint as pp

data = load_diabetes()

print(data)
df = pd.DataFrame(data.data, columns=data.feature_names)

print(df)
X = df
y = data.target  # Target

y_Scaler = StandardScaler()
y = y_Scaler.fit_transform(y.reshape(-1, 1))
# print(X)
# print(y)
model = IGANN(task="regression", verbose=0)

model.fit(X, y)

GAM = GAMmodel(model, task="regression")

pp(GAM.set_shape_functions())

y_pred = GAM.predict_raw(X)

y_pred_true = model.predict(X)

print(y_pred_true)
print(y_pred)


# %%
