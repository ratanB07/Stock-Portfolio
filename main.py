from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings

# Filter out the specific FutureWarning about 'M' deprecation
warnings.filterwarnings('ignore', category=FutureWarning, message="'M' is deprecated")

app = Flask(__name__)

# Define the path to your folder containing the stock data files
data_folder = r"C:\Users\Ratan Biswakarmakar\Desktop\SB Work\stock_data_1h\stock_data_1h"

# Read all the CSV files in the folder and store them in a dictionary
stock_data = {}

# List all files in the directory
files = os.listdir(data_folder)

for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(data_folder, file)
        stock_name = file.split('.')[0]
        try:
            df = pd.read_csv(file_path, index_col="datetime", parse_dates=["datetime"])
            stock_data[stock_name] = df["close"]
        except Exception as e:
            print(f"Error reading {file}: {e}")

# Concatenate all stock data into a single DataFrame
data = pd.concat(stock_data.values(), axis=1)
data.columns = stock_data.keys()

def calculate_portfolio_with_xgboost(stock_symbol, monthly_investment, years=5):
    prices = data[stock_symbol]
    # Use 'ME' instead of 'M' for month end frequency
    prices = prices.resample('ME').mean()
    prices = prices.dropna()
    prices = prices[~prices.isin([np.inf, -np.inf])]

    features = []
    target = []
    
    for i in range(1, len(prices)-1):
        features.append([prices.iloc[i-1], prices.iloc[i]])
        target.append(prices.iloc[i+1] / prices.iloc[i] - 1)
    
    X = np.array(features)
    y = np.array(target)
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Target contains NaN or Inf values.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', early_stopping_rounds=10)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    predicted_returns = model.predict(X_test)
    predicted_returns = np.nan_to_num(predicted_returns, nan=0.0, posinf=0.0, neginf=0.0)
    cumulative_returns = np.cumsum(predicted_returns)
    
    total_investment = monthly_investment * years * 12
    final_value = total_investment * (1 + cumulative_returns[-1])
    
    return final_value, total_investment, cumulative_returns

def generate_chart(stock_results):
    # Create a new figure for each chart
    plt.clf()
    plt.figure(figsize=(12, 8))
    
    for stock, result in stock_results.items():
        plt.plot(result['cumulative_returns'], label=stock)
    
    plt.title("Portfolio Cumulative Returns Over Time")
    plt.xlabel("Months")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close('all')  # Close all figures to free up memory
    buf.close()
    return chart

@app.route('/', methods=['GET', 'POST'])
def home():
    portfolio_results = None
    chart = None
    error = None

    if request.method == 'POST':
        try:
            selected_stocks = request.form.getlist('stock_symbols')
            monthly_investment = float(request.form['investment_amount'])
            years = int(request.form['years'])
            
            portfolio_results = {}
            total_portfolio_value = 0
            total_investment = 0
            
            for stock in selected_stocks:
                final_value, stock_investment, cumulative_returns = calculate_portfolio_with_xgboost(
                    stock, monthly_investment, years
                )
                portfolio_results[stock] = {
                    'final_value': final_value,
                    'investment': stock_investment,
                    'profit': final_value - stock_investment,
                    'cumulative_returns': cumulative_returns
                }
                total_portfolio_value += final_value
                total_investment += stock_investment
            
            portfolio_results['total'] = {
                'final_value': total_portfolio_value,
                'investment': total_investment,
                'profit': total_portfolio_value - total_investment
            }
            
            chart = generate_chart({k: v for k, v in portfolio_results.items() if k != 'total'})
            
        except Exception as e:
            error = str(e)

    return render_template('index.html',
                         portfolio_results=portfolio_results,
                         chart=chart,
                         error=error,
                         stocks=data.columns)

if __name__ == "__main__":
    app.run(debug=True)