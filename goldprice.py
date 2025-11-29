import pandas as pd
import yfinance as yf
import duckdb
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")


from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')

ticker = config.get('SETTINGS', 'STOCK_TICKER')
lookback_window_years = config.getint('SETTINGS', 'DATA_WINDOW_YEARS')
prediction_days = config.getint('SETTINGS', 'PREDICTION_DAYS')
db_file = config.get('SETTINGS', 'DB_FILE')
db_table_name = config.get('SETTINGS', 'DB_TABLE_NAME')

regressors = []
for key in config['REGRESSORS']:
    regressors.append(config.get('REGRESSORS', key))

@st.cache_resource
def get_db_connection(db_file=db_file):
    # Initialize database connection
    db_connection = duckdb.connect(database=db_file, read_only=False)
    if db_connection is None:
        raise Exception("Failed to connect to the database.")
    
    # Create table if it doesn't exist
    db_connection.execute(f"""
        CREATE TABLE IF NOT EXISTS {db_table_name} (
            ds DATE PRIMARY KEY,
            y DOUBLE,
        );
    """)
    return db_connection

class DataService:
    # Handle data fetching, storage and retrieval
    def __init__(self):
        self.db_conn = get_db_connection()
        self.fetch_and_prepare_data()
    
    def get_latest_db_date(self):
        # Get the latest date stored in the database
        result = self.db_conn.execute(f"SELECT MAX(ds) FROM {db_table_name};").fetchone()

        if result and result[0]:
            # Return the latest date from the database
            return pd.to_datetime(result[0])
        else:
            start_date = pd.Timestamp.now() - pd.DateOffset(years=lookback_window_years)
            return start_date
    
    def fetch_new_data_from_yfinance(self, start_date: pd.Timestamp):
        # Fetch latest data from yfinance
        fetch_end_date = pd.Timestamp.now()
        fetch_start_date = (start_date + timedelta(days=1))
        
        print(f"Fetching data from {fetch_start_date} to {fetch_end_date}")

        # if (fetch_start_date >= fetch_end_date):
        #     print("Data is already up to date.")
        #     return None

        try:
            tickers = [ticker] + regressors
            data = yf.download(
                tickers, 
                start=fetch_start_date, 
                end=fetch_end_date,
                interval='1d'
                )
            

            if data.empty:
                print("No new data fetched from yfinance.")
                return pd.DataFrame(columns=['ds', 'y'])
            
            data.reset_index(inplace=True)
            print(data.head())
            new_data = data.iloc[:, [0, 1]]
            new_data.columns = ['ds', 'y']


            # print(new_data.columns)
            # print(new_data.head())


            # Removing overlapping data
            new_data = new_data[new_data['ds'] >= fetch_start_date]

            print(f"Fetched {len(new_data)} new records.")
            return new_data

        except Exception as e:
            raise Exception(f"Error fetching data from yfinance: {e}")
        
    def fetch_and_prepare_data(self):
        # Fetch new data and store in database

        # Get latest date from database
        latest_db_date = self.get_latest_db_date()
        new_data_df = self.fetch_new_data_from_yfinance(latest_db_date)

        if not new_data_df.empty:
            print(f"Adding {len(new_data_df)} new data points to the database...")
            self.db_conn.from_df(new_data_df).insert_into(db_table_name)
            print("Data added to the database.")

        historical_data = self.db_conn.execute(f"SELECT ds, y FROM {db_table_name} ORDER BY ds ASC;").fetchdf()

        if historical_data.empty:
            raise Exception("No data available in the database after fetching. Cannot train a model...")
            return None
        
        historical_data['ds'] = pd.to_datetime(historical_data['ds'])
        return historical_data

@st.cache_data(
    show_spinner="Training prediction model...", 
    hash_funcs={pd.DataFrame: lambda df: hash(df.to_json())}
)
def _cache_train_and_predict(_data: pd.DataFrame, prediction_days: int):
    model = Prophet(
        growth='linear',
        yearly_seasonality='auto',
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )

    model.fit(_data)
    future = model.make_future_dataframe(periods=prediction_days)
    forecast = model.predict(future)
    return model, forecast

class ModelService:
    # Handle model training and prediction
    def train_and_predict(self, data: pd.DataFrame, prediction_days: int):
        return _cache_train_and_predict(data, prediction_days)


def plot_forecast(m, forecast, title_suffix):
    """Generates an interactive Plotly chart for the forecast."""
    fig = plot_plotly(m, forecast)
    
    fig.update_layout(
        title={'text': f'Gold Price ({ticker}) Forecast and Historical Data: {title_suffix}',
               'x': 0.5, 'xanchor': 'center'},
        yaxis_title="Closing Price (USD)",
        xaxis_title="Date",
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Customizing trace colors and names
    fig.data[0].name = 'Historical Price'
    fig.data[1].name = 'Prediction Interval'
    fig.data[2].name = 'Forecast Price'
    fig.data[2].line.color = 'rgb(255, 165, 0)'
    
    st.plotly_chart(fig, use_container_width=True)

def display_metrics(forecast, forecast_days):
    """Displays key prediction metrics using Streamlit's st.metric."""
    
    # Check if forecast has enough data points (historical + forecast)
    if len(forecast) < forecast_days + 1:
        st.warning("Insufficient data for detailed metrics.")
        return

    # Determine indices dynamically based on the length of the forecast part
    # We look for the last date in the forecast that has a non-zero y (original historical data)
    # Since we combine historical and future data, we need to find the split point.
    
    # Find the index where 'yhat' is the prediction for the day after the last historical day
    # A simple way: the index of the last historical data point is len(forecast) - forecast_days - 1
    
    last_hist_idx = len(forecast) - forecast_days - 1
    
    # Handle the case where the forecast is too small or index is out of bounds
    if last_hist_idx < 0:
        st.error("Error calculating historical metrics indices.")
        return

    first_forecast_idx = len(forecast) - forecast_days
    final_forecast_idx = len(forecast) - 1

    last_historical_date = forecast.iloc[last_hist_idx]['ds']
    last_historical_price = forecast.iloc[last_hist_idx]['yhat'] # yhat on historical data is often close to y
    
    first_forecast_date = forecast.iloc[first_forecast_idx]['ds']
    first_forecast_price = forecast.iloc[first_forecast_idx]['yhat']
    
    final_forecast_date = forecast.iloc[final_forecast_idx]['ds']
    final_forecast_price = forecast.iloc[final_forecast_idx]['yhat']

    # Calculate the projected change over the forecast period
    projected_change = ((final_forecast_price - first_forecast_price) / first_forecast_price) * 100

    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        label="Last Historical Close (USD)", 
        value=f"${last_historical_price:,.2f}",
        delta=f"As of {last_historical_date}",
        delta_color="off"
    )

    col2.metric(
        label=f"Next Day's Prediction (USD)",
        value=f"${first_forecast_price:,.2f}",
        delta=f"Predicted for {first_forecast_date}",
        delta_color="off"
    )

    col3.metric(
        label=f"{forecast_days}-Day Projected Change",
        value=f"{projected_change:.2f}%",
        delta=f"Target: ${final_forecast_price:,.2f} on {final_forecast_date}",
        delta_color="inverse" if projected_change < 0 else "normal"
    )
    
def app():
    """The main Streamlit application function."""
    st.set_page_config(
        page_title="Gold Price Forecasting System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💰 Gold Price Forecasting System")
    st.markdown("Forecasting the price of the Gold Futures Contract (`GC=F`) using a **Prophet** time series model.")

    # State initialization
    if 'forecast_days' not in st.session_state:
        st.session_state.forecast_days = prediction_days
    
    # --- Sidebar for Controls ---
    with st.sidebar:
        st.header("System Controls")
        
        forecast_days_input = st.slider(
            "Select Forecast Period (Days)",
            min_value=7,
            max_value=180,
            value=st.session_state.forecast_days,
            step=7,
            key='days_slider'
        )
        st.session_state.forecast_days = forecast_days_input
        
        st.markdown("---")
        st.subheader("Database & Data Source")
        # Update sidebar text to reflect the pure in-memory usage
        st.markdown(f"**Ticker:** `{ticker}`")
        st.markdown(f"**Data Strategy:** Incremental Fetch")
        st.markdown(f"**Database:** DuckDB (Pure In-Memory)")
        
        if st.button("Clear Cache and Retrain Model", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
            
    # --- Orchestration ---
    
    # 1. Initialize Services
    data_service = DataService()
    model_service = ModelService()
    
    # 2. Data Acquisition (Incremental Load)
    # This function now handles the incremental update and returns the full dataset
    historical_data = data_service.fetch_and_prepare_data()
    
    if historical_data is not None and not historical_data.empty:
        
        # 3. Training and Prediction (Cached)
        m, forecast = model_service.train_and_predict(historical_data, st.session_state.forecast_days)
        
        # 4. Display Results
        
        st.subheader("Forecasting Summary")
        display_metrics(forecast, st.session_state.forecast_days)
        
        st.subheader(f"Interactive Forecast Chart ({st.session_state.forecast_days} Days)")
        plot_forecast(m, forecast, f"{st.session_state.forecast_days}-Day Outlook")
        
        st.subheader("Model Components: Trend and Seasonality")
        st.markdown("Prophet decomposes the time series into components (Trend, Yearly, and Weekly) to show how they contribute to the final forecast. [Image of Time Series Decomposition]")
        
        # Plotting model components
        fig_components = m.plot_components(forecast)
        st.pyplot(fig_components, use_container_width=True)
        plt.close(fig_components) 
        
        # Display Raw Data Tables
        col_hist, col_forecast = st.columns(2)
        with col_hist:
            st.subheader("Historical Data Snippet (from DB)")
            st.dataframe(historical_data.tail(10).set_index('ds'), use_container_width=True)
            
        with col_forecast:
            st.subheader(f"Next {st.session_state.forecast_days} Day Forecast")
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(st.session_state.forecast_days)
            forecast_df.columns = ['Date', 'Predicted Close', 'Lower Bound', 'Upper Bound']
            st.dataframe(
                forecast_df.set_index('Date'), 
                use_container_width=True, 
                column_config={"Predicted Close": st.column_config.NumberColumn(format="$%.2f")}
            )

if __name__ == "__main__":
    app()