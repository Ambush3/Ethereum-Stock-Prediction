# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from autots import AutoTS

# Plotting Current and Past Ethereum Prices
sns.set()
plt.style.use('seaborn-darkgrid')

data = pd.read_csv("F:/PyCharm/Machine-Learning/Ethereum Stock Prediction/ETH-USD.csv")
print(data.head())

# Dropping blank data
data.dropna()

# Plotting Ethereum Prices
plt.figure(figsize=(10, 4))
plt.title("Ethereum Price")
plt.xlabel("Date")
plt.ylabel("Close")
plt.plot(data["Close"])
plt.show()


# Forecasting Future Ethereum Prices
model = AutoTS(forecast_length=10, frequency='infer', ensemble='simple', drop_data_older_than_periods=200)
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)

prediction = model.predict()
forecast = prediction.forecast
print("Ethereum Price Prediction")
print(forecast)