import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import os
import plotly.graph_objects as go
import glob

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from arch import arch_model


st.set_page_config(layout="wide")

def convert_to_numeric(value):
    if isinstance(value, str):  
        value = value.replace('$', '').replace(',', '')  # Remove the dollar sign and commas
        if 'B' in value:
            return float(value.replace('B', '')) * 1e9  # Convert billions to numeric
        elif 'M' in value:
            return float(value.replace('M', '')) * 1e6  # Convert millions to numeric
        elif 'K' in value:
            return float(value.replace('K', '')) * 1e3  # Convert thousands to numeric
    return float(value)  

df_leader = pd.read_csv("archive/Current Crypto leaderboard.csv")

def summary_page():
    st.write("*An OKX assessment by Ben Cheong*")
    st.title("Summarized Cryptocurrency Analysis :coin:")

    st.header("1. Market Overview", divider=True)

    df_leader['Market Cap'] = df_leader['Market Cap'].apply(convert_to_numeric)
    df_leader['Vol (24H)'] = df_leader['Vol (24H)'].apply(convert_to_numeric)

    df_leader['Chg (24H)'] = df_leader['Chg (24H)'].str.replace('%', '').astype(float)
    df_leader['Chg (7D)'] = df_leader['Chg (7D)'].str.replace('%', '').astype(float)

    df_leader.index += 1

    st.write("Top 100 Cryptocurrencies Data:")
    st.write(df_leader)
   
    df_leader['Percentage'] = (df_leader['Market Cap'] / df_leader['Market Cap'].sum()) * 100
    df_leader['Market Percentage'] = df_leader['Name'] + ' (' + df_leader['Percentage'].round(2).astype(str) + '%)'

    fig = px.pie(df_leader, values='Market Cap', names='Name', title='Market Capitalization Distribution:',
                    hover_data={'Market Cap': ':.2f'}, labels={'Market Cap': 'Market Cap (USD)'})

    fig.update_layout(width=1000, height=800)

    fig.update_traces(textposition='inside', textinfo='percent+label')

    st.plotly_chart(fig)
    st.write(df_leader['Market Percentage'])

    total_market_cap = df_leader['Market Cap'].sum()
    st.markdown(f"- The total market capitalization of the top 100 cryptocurrencies currently stands at **${total_market_cap:.2f}**, showcasing the significant size of the cryptocurrency market.")
    st.markdown(f"- The top 5 cryptocurrencies account for approximately **{df_leader['Percentage'][:5].sum():.2f}%** of the total market capitalization.")

    top_10 = df_leader.sort_values(by='Market Cap', ascending=False).head(10)

    top_10 = df_leader.sort_values(by='Market Cap', ascending=False).head(10)

    fig_2 = px.scatter(top_10, 
                    x='Market Cap', 
                    y='Total Vol', 
                    color='Name', 
                    text='Name',  
                    title="Top 10 Cryptocurrencies: Market Cap vs Total Volume",
                    labels={"Market Cap": "Market Cap (USD)", "Total Vol": "Total Volume (%)"},
                    hover_name="Name", 
                    hover_data=["Price (USD)", "Chg (24H)", "Chg (7D)"])  # Additional data on hover

    fig_2.update_traces(marker=dict(size=10, opacity=1), text=None)

    fig_2.update_layout(width=1000, height=600)
    st.plotly_chart(fig_2)

    st.markdown("- X-axis (Market Cap): Represents the total market capitalization of each cryptocurrency. Higher values on the right indicate cryptocurrencies with a larger market share.")
    st.markdown("- Y-axis (Total Volume as %): Shows the volume as a percentage of the total tradable volume for that particular currency.")
    st.markdown("- This graph serves to show that while huge coins like Bitoin and Ethereum that have high market caps with significant, but not necessarily the highest, trading volumes, suggests that these cryptocurrencies are widely held but may not be the most frequently traded on a daily basis.")

    st.header("2. Lifetime Price Performance", divider=True)

    performance_list = []

    for file in os.listdir("archive/Top 100 Crypto Coins"):
        df = pd.read_csv(os.path.join("archive/Top 100 Crypto Coins", file))
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        df = df.sort_values(by='Date')
        
        if len(df) > 1:
            first_non_zero_price = df[df['Close'] > 0]['Close'].iloc[0] if not df[df['Close'] > 0].empty else 0
            if first_non_zero_price == 0:
                continue
            
            last_price = df['Close'].iloc[-1]
            
            price_performance = round(((last_price - first_non_zero_price) / first_non_zero_price) * 100)

            # Get the currency name from the CSV or file name (assuming it's included in the file or file name)
            currency_name = file.split('/')[-1].replace('.csv', '')
            
            performance_list.append({
                'Currency': currency_name,
                'First Close Price': first_non_zero_price,
                'Latest Close Price': last_price,
                'Price Performance (%)': price_performance
            })

    lifetime_performance_df = pd.DataFrame(performance_list)

    lifetime_performance_df = lifetime_performance_df.sort_values(by='Price Performance (%)', ascending=False)

    lifetime_performance_df.reset_index(drop=True, inplace=True)

    lifetime_performance_df.index += 1

    st.write("*Double click on cell to display exact value*")
    st.write(lifetime_performance_df)

    st.markdown("- Lifetime Price Performance refers to the percentage change in the price of a cryptocurrency from the first recorded price (starting price) to the most recent price (closing price) within the available data. It is calculated as:")
    st.latex(r'''\text{Lifetime Price Performance} = \frac{\text{Latest Close Price} - \text{First Close Price}}{\text{First Close Price}} \times 100''')
    st.markdown("- This metric shows how much the price of a cryptocurrency has increased or decreased over its entire lifespan, according to the data available.")
    st.markdown("- It is able to provide simple summarized insights into the long-term performance of the cryptocurrency, helping investors understand whether it has gained or lost value over time.")

    st.header("3. Volatility Analysis", divider=True)
    st.write("*Click to show/hide currencies*")

    csv_files = glob.glob("archive/Top 100 Crypto Coins/*.csv")

    volatility_list = []

    with st.spinner('Loading data...'):
        for file in csv_files:
            df = pd.read_csv(file)
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')
            
            # Extract cryptocurrency name from the file name
            crypto_name = file.split('/')[-1].replace('.csv', '')
            
            # Check if there are enough rows to calculate volatility
            if len(df) > 1:
                # Calculate daily log returns
                df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))
                
                # Calculate rolling 30-day volatility (standard deviation of log returns)
                df['Rolling Volatility'] = df['Log Return'].rolling(window=30).std()
                
                # Store each cryptocurrency's rolling volatility with its dates and crypto name
                for index, row in df.iterrows():
                    volatility_list.append({
                        'Crypto': crypto_name,
                        'Date': row['Date'],
                        'Rolling Volatility': row['Rolling Volatility']
                    })

        volatility_df = pd.DataFrame(volatility_list)

        fig_3 = go.Figure()

        for crypto in volatility_df['Crypto'].unique():
            crypto_data = volatility_df[volatility_df['Crypto'] == crypto]
            fig_3.add_trace(go.Scatter(
                x=crypto_data['Date'], 
                y=crypto_data['Rolling Volatility'],
                mode='lines',
                name=crypto,
                visible='legendonly' if crypto != volatility_df['Crypto'].unique()[0] else True  # Show only the first crypto by default
            ))

        fig_3.update_layout(
            title="Cryptocurrency Rolling Volatility (30-day Window)",
            xaxis_title="Date",
            yaxis_title="Rolling Volatility",
            height=750,
            width=1300
        )
        st.plotly_chart(fig_3)

    st.markdown("- The rolling volatility graph measures the standard deviation of price movements (log returns) over the past 30 days. This tells you how much the price has been moving up or down during that period.")
    st.markdown("- High rolling volatility suggests more significant price swings, while low rolling volatility indicates relatively stable prices.")
    st.markdown("- If the 30-day rolling volatility is high, it indicates that the asset has been riskier or more volatile in the past month (30 days).")
    st.markdown("- Sudden spikes in volatility can indicate the occurrence of trends in market events, such as news that affect the price of the cryptocurrency, large trades, market manipulation, or changes in market sentiment and even political events affecting the market.")
    st.markdown("- By comparing the rolling volatility of multiple cryptocurrencies, we can see which assets are more volatile than others in the short term. This is useful for comparing risk profiles across different cryptocurrencies.")
    
    st.header("4. Correlation Analysis", divider=True)

    csv_files = [
        'bitcoin.csv', 'ethereum.csv', 'tether.csv', 'usd coin.csv',
        'BNB.csv', 'Binance USD.csv', 'xrp.csv', 'cardano.csv',
        'solana.csv', 'dogecoin.csv'
    ]

    prices = pd.DataFrame()

    for file in csv_files:
        crypto_name = file.replace('.csv', '').replace(' ', '_')  # Extract crypto name from the file name, replace spaces with underscores
        df = pd.read_csv(os.path.join("archive/Top 100 Crypto Coins", file))
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        df = df.sort_values(by='Date')
        
        if prices.empty:
            prices = df[['Date', 'Close']].rename(columns={'Close': crypto_name})
        else:
            prices = prices.merge(df[['Date', 'Close']].rename(columns={'Close': crypto_name}), on='Date')

    daily_returns = prices.set_index('Date').pct_change()

    correlation_matrix = daily_returns.corr()

    fig_4 = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,  # Correlation matrix values
        x=correlation_matrix.columns,  # Cryptocurrency names on the x-axis
        y=correlation_matrix.columns,  # Cryptocurrency names on the y-axis
        text=correlation_matrix.values,  # Values to display on the heatmap
        texttemplate="%{text:.2f}",  # Display the values with 2 decimal places
        colorscale=[  # Custom colorscale: blue for -1, white for 0, red for +1
            [0, "blue"],   # -1 correlation (strong negative)
            [0.5, "white"],  # 0 correlation (no correlation)
            [1, "red"]     # +1 correlation (strong positive)
        ],
        zmin=-1,  # Minimum value (-1 correlation)
        zmax=1,   # Maximum value (+1 correlation)
        colorbar=dict(title="Correlation")  # Label for the color bar
    ))

    fig_4.update_layout(
        title="Correlation Between Cryptocurrencies",
        xaxis_title="Cryptocurrency",
        yaxis_title="Cryptocurrency",
        height=800,
        width=1000
    )

    st.plotly_chart(fig_4)

    st.markdown("The correlation matrix shown in your heatmap represents the relationship between the daily returns of the top cryptocurrencies over the given time period. Each value in the matrix shows how closely the returns of two cryptocurrencies are correlated with each other, with values ranging from -1 to 1:")
    st.markdown("- 1: Perfect positive correlation – When one cryptocurrency's price increases, the other tends to increase in the same direction.")
    st.markdown("- 0: No correlation – The prices of the two cryptocurrencies move independently of each other.")
    st.markdown("- -1: Perfect negative correlation – When one cryptocurrency's price increases, the other tends to decrease in the opposite direction.")
    st.markdown("The correlation matrix is calculated by taking the **daily percentage change** in the closing prices of each cryptocurrency, then using **Pearson correlation** to measure how linearly related the returns are between each pair of cryptocurrencies:")
    st.latex(r'''\text{Correlation} = \frac{\text{Covariance}(X, Y)}{\text{Std Dev}(X) \times \text{Std Dev}(Y)}''')
    st.markdown("- It can be seen that Bitcoin (BTC) and Ethereum (ETH) show a strong positive correlation of 0.74, meaning they tend to move in the same direction. This is expected, as both are major assets in the cryptocurrency market and often react similarly to market conditions.")
    st.markdown("- It is also seen that Tether (USDT) and USD Coin (USDC), both stablecoins, show a strong negative correlation of -0.59. This might suggest that these stablecoins are inversely related in terms of market reactions.")
    st.markdown("- Dogecoin shows relatively low correlation with the other top cryptocurrencies (mostly between 0.1 and 0.3), indicating that its price movements are more independent and driven by unique factors, such as social media (Elon Musk's social media influence) or specific market events.")


def individual_page():
    st.write("*An OKX assessment by Ben Cheong*")
    st.title("Individual Cryptocurrency Visualization :chart_with_upwards_trend:")

    folder_path = "archive/Top 100 Crypto Coins/"  
    csv_files = glob.glob(f"{folder_path}/*.csv")

    cryptos = [file.split('/')[-1].replace('.csv', '').replace(' ', '_') for file in csv_files]

    selected_crypto = st.selectbox('Select a Cryptocurrency', cryptos)

    selected_file = f"{folder_path}/{selected_crypto.replace('_', ' ')}.csv"  
    df = pd.read_csv(selected_file)

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values(by='Date')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Open'],
        mode='lines',
        name='Open Price',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f"Lifetime Open and Close Prices for {selected_crypto.replace('_', ' ')}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=600
    )
    
    df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))  # Calculate the daily log return using NumPy
    df['30-Day Rolling Volatility'] = df['Log Return'].rolling(window=30).std()  # Annualized volatility

    fig_volatility = go.Figure()

    fig_volatility.add_trace(go.Scatter(
        x=df['Date'],
        y=df['30-Day Rolling Volatility'],
        mode='lines',
        name='30-Day Rolling Volatility',
        line=dict(color='green')
    ))

    fig_volatility.update_layout(
        title=f"30-Day Rolling Volatility for {selected_crypto.replace('_', ' ')}",
        xaxis_title="Date",
        yaxis_title="30-Day Rolling Volatility",
        height=600
    )

    st.plotly_chart(fig)
    st.plotly_chart(fig_volatility)


def prediction_page():

    st.write("*An OKX assessment by Ben Cheong*")
    st.title("Cryptocurrency Volatility Prediction :robot_face:")
    st.header("Random Forest Regressor", divider=True)

    def calculate_annualized_volatility(df):
        # Calculate daily log returns
        df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Rolling 30-day average return and variance
        rolling_mean = df['Log Return'].rolling(window=30).mean()
        rolling_var = df['Log Return'].rolling(window=30).var()
        
        # Annualized volatility calculation
        df['Annualized Volatility'] = np.sqrt(365) * np.sqrt(rolling_var)
        
        return df

    def prepare_features(df):
        # Create features for the model
        df['Price Range'] = df['High'] - df['Low']
        df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        df = df.dropna()
        
        features = df[['Price Range', 'Volume', 'Log Return']]
        target = df['Annualized Volatility']
        
        return features, target

    folder_path = "archive/Top 100 Crypto Coins/"
    csv_files = glob.glob(f"{folder_path}/*.csv")
    cryptos = [file.split('/')[-1].replace('.csv', '').replace(' ', '_') for file in csv_files]

    selected_crypto = st.selectbox('Select a Cryptocurrency', cryptos)
    selected_file = f"{folder_path}/{selected_crypto.replace('_', ' ')}.csv"
    df = pd.read_csv(selected_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df = calculate_annualized_volatility(df)

    features, target = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=target,
        mode='lines',
        name='Actual Volatility'
    ))

    predicted_volatility = model.predict(features)
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=predicted_volatility,
        mode='lines',
        name='Predicted Volatility',
        line=dict(color='red')
    ))

    fig.add_annotation(
        x=df['Date'].iloc[-1],  
        y=max(target),  
        text=f"Mean Squared Error (MSE): {mse:.4f}",
        showarrow=False,
        font=dict(size=18, color="red"),
        align="center",
        borderwidth=2,
        borderpad=4,
        bgcolor="white",
        opacity=0.8
    )

    fig.update_layout(
        title=f"Actual vs Predicted Annualized Volatility for {selected_crypto.replace('_', ' ')}",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        height=800,
        width=1600
    )

    st.plotly_chart(fig)

    st.write("The prediction model for this use case is Random Forest Regressor. It is an ensemble learning method that combines multiple decision trees to make predictions. It improves the accuracy and robustness of the model by averaging the predictions of several decision trees, which helps reduce overfitting and variance compared to using a single tree.")
    st.write("In this scenario, the features selected are: **'Price Range', 'Volume', and 'Log Return' (Calculated from formula)** and are used to predict the target: **'Annualized Volatility'** of the cryptocurrency.")
    st.write("Mean Squared Error (MSE) is a metric used to evaluate the performance of the model. It measures the average squared difference between the actual and predicted values. A lower MSE indicates a better fit of the model to the data.")
    st.write("In summary, the chosen model: **Random Forest Regressor** learns from past values of log returns, price range, and volume to understand how they relate to volatility.")

    st.header("GARCH", divider=True)

    def calculate_lifetime_annualized_volatility(df):
        df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()  
        rolling_volatility = df['Log Return'].rolling(window=30).std()

        annualized_volatility = rolling_volatility * np.sqrt(365)
        
        df.loc[:, 'Annualized Volatility'] = annualized_volatility
        return df

    def predict_garch_volatility(df, forecast_days=30):
        df['Log Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()  

        # Fit a GARCH(1,1) model to the log returns
        model = arch_model(df['Log Return'], vol='Garch', p=1, q=1)  
        model_fit = model.fit(disp="off")  

        # Predict future volatility (e.g., next 'forecast_days' days)
        forecast = model_fit.forecast(horizon=forecast_days)

        predicted_variance = forecast.variance.iloc[-1]
        predicted_volatility = np.sqrt(predicted_variance)

        # Convert daily volatility to annualized volatility
        predicted_annualized_volatility = predicted_volatility * np.sqrt(365)

        return predicted_annualized_volatility

    selected_crypto = st.selectbox('Select a Cryptocurrency', cryptos, key="crypto_select")

    def load_crypto_data(file_name):
        selected_file = f"archive/Top 100 Crypto Coins/{file_name.replace('_', ' ')}.csv"
        df = pd.read_csv(selected_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df

    df = load_crypto_data(selected_crypto)

    df = calculate_lifetime_annualized_volatility(df)

    months = st.slider('Select Number of Months for Forecast', 1, 12, 3, key="months_slider")
    forecast_days = months * 30  # Convert months to days (approximation)

    predicted_volatility = predict_garch_volatility(df, forecast_days=forecast_days)

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq='D')[1:]

    lifetime_volatility_dates = df.index
    lifetime_volatility_values = df['Annualized Volatility']

    predicted_volatility_dates = future_dates
    predicted_volatility_values = [lifetime_volatility_values.iloc[-1]] + predicted_volatility.tolist()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=lifetime_volatility_dates,
        y=lifetime_volatility_values,
        mode='lines',
        name='Lifetime Annualized Volatility',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=predicted_volatility_dates,
        y=predicted_volatility_values,
        mode='lines',
        name='Predicted Annualized Volatility',
        line=dict(color='green')
    ))

    fig.update_layout(
        title=f"Lifetime and Predicted Annualized Volatility for {selected_crypto.split('.')[0].capitalize()}",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        height=800,
        width=1600
    )

    st.plotly_chart(fig)

    st.write("Unlike the Random Forest Regressor, GARCH is a time-series model specifically designed to model and **forcast** volatility in time series data.")
    st.write("GARCH (Generalized Autoregressive Conditional Heteroskedasticity) is a widely used statistical model for predicting and modeling time-varying volatility in financial time series data, such as stock prices, returns, or exchange rates.")
    st.write("GARCH captures volatility patterns by assuming that volatility at any point in time depends on: 1) the past volatility, 2) the past squared returns, and 3) the past squared forecast errors.")
    st.write("Features used by GARCH:")
    st.markdown("- Log Returns: GARCH models the log returns of crypto prices")
    st.markdown("- Past Volatility: The model uses past volatility (the conditional variance) as an input to forecast future volatility.")
    st.markdown("- Residuals: GARCH uses past residuals, which are the differences between the actual returns and the expected returns, to measure how volatile the market was in the past.")

page = st.sidebar.selectbox("Page Select:", ("1. Summarized Analysis", "2. Individual Crypto Chart", "3. Volatility Prediction"))

if page == "1. Summarized Analysis":
    summary_page()
elif page == "2. Individual Crypto Chart":
    individual_page()
elif page == "3. Volatility Prediction":
    prediction_page()
