import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Load the cleaned data
data = pd.read_csv('/Users/anithasmac/PycharmProjects/FinalProject/Model_Data.csv')

store_options = data['Store'].unique()
dept_options = data['Dept'].unique()
type_options = data['Type'].unique()
size_options = data['Size'].unique()

# Load the models
with open('XGBoost.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('XGBoostNOMarkdown.pkl', 'rb') as file:
    loaded_model_no_markdown = pickle.load(file)

# Set page configuration
st.set_page_config(
    page_title="Weekly Sales Prediction",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar Menu
with st.sidebar:
    selected = option_menu("Menu", ["Price Prediction", "Impact Analysis"], 
        icons=['currency-exchange', 'bar-chart-line'], menu_icon="menu-button"
    )

# Custom CSS 
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        h1 {
            color: #4CAF50;
            text-align: center;
            font-size: 3em;
        }
        .stSidebar {
            background-color: #2c3e50;
        }
        .sidebar-content {
            font-size: 18px;
            color: white;
        }
        footer {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

if selected == 'Price Prediction':
    st.title("Sales Forecast")
    tab1, tab2 = st.tabs(["Predict with Markdowns", "Predict without Markdowns"])
    
    with tab1:
        # Input form for predictions
        st.subheader("Enter Details:")
        store = st.selectbox("Select Store", options=store_options, key='tab1_store')
        dept = st.selectbox("Select Department", options=dept_options, key='tab1_dept')
        is_holiday = st.checkbox("Is Holiday?", value=False, key='tab1_holiday')
        temperature = st.number_input("Temperature (°F)", value=60.0, key='tab1_temp')
        fuel_price = st.number_input("Fuel Price ($)", value=3.0, key='tab1_fuel')
        cpi = st.number_input("CPI", value=220.0, key='tab1_cpi')
        unemployment = st.number_input("Unemployment Rate (%)", value=7.0, key='tab1_unemp')
        store_type = st.selectbox("Select Type", options=type_options, key='tab1_type')
        store_size = st.selectbox("Select Size of the store", options=size_options, key='tab1_size')
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2023, step=1, key='tab1_year')

        # Lagged Sales and Holiday Inputs
        lagged_sales_1_weeks = st.number_input("Lagged Sales (1 week ago)", value=0.0, step=10.0, key='tab1_lag1')
        lagged_sales_2_weeks = st.number_input("Lagged Sales (2 weeks ago)", value=0.0, step=10.0, key='tab1_lag2')
        lagged_sales_4_weeks = st.number_input("Lagged Sales (4 weeks ago)", value=0.0, step=10.0, key='tab1_lag4')
        lagged_sales_12_weeks = st.number_input("Lagged Sales (12 weeks ago)", value=0.0, step=10.0, key='tab1_lag12')
        is_holiday_week = st.checkbox("Is Holiday Week?", value=False, key='tab1_holidayweek')
        lagged_holiday_1_week = st.checkbox("Lagged Holiday (1 week ago)?", value=False, key='tab1_hol1')
        lagged_holiday_2_weeks = st.checkbox("Lagged Holiday (2 weeks ago)?", value=False, key='tab1_hol2')
        markdown1 = st.number_input("MarkDown1", value=0.0, step=10.0, key='tab1_mark1')
        markdown2 = st.number_input("MarkDown2", value=0.0, step=10.0, key='tab1_mark2')
        markdown3 = st.number_input("MarkDown3", value=0.0, step=10.0, key='tab1_mark3')
        markdown4 = st.number_input("MarkDown4", value=0.0, step=10.0, key='tab1_mark4')
        markdown5 = st.number_input("MarkDown5", value=0.0, step=10.0, key='tab1_mark5')

        # Predict button and prediction
        if st.button("Predict Weekly Sales", key='tab1'):
            input_data = np.array([[store, dept, int(is_holiday), temperature, fuel_price, cpi, unemployment, 
                                    store_type, store_size, lagged_sales_1_weeks, lagged_sales_2_weeks, 
                                    lagged_sales_4_weeks, lagged_sales_12_weeks, 
                                    int(is_holiday_week), int(lagged_holiday_1_week), 
                                    int(lagged_holiday_2_weeks), year,
                                    markdown1, markdown2, markdown3, markdown4, markdown5]])
            prediction = loaded_model.predict(input_data)
            st.session_state['prediction'] = prediction
            st.session_state['markdowns'] = 'Yes'
            st.success(f"Predicted Weekly Sales: ${prediction[0]:,.2f}")
    
    with tab2:
        # Input form for predictions
        st.subheader("Enter Details:")
        store = st.selectbox("Select Store", options=store_options, key='tab2_store')
        dept = st.selectbox("Select Department", options=dept_options, key='tab2_dept')
        is_holiday = st.checkbox("Is Holiday?", value=False, key='tab2_holiday')
        temperature = st.number_input("Temperature (°F)", value=60.0, key='tab2_temp')
        fuel_price = st.number_input("Fuel Price ($)", value=3.0, key='tab2_fuel')
        cpi = st.number_input("CPI", value=220.0, key='tab2_cpi')
        unemployment = st.number_input("Unemployment Rate (%)", value=7.0, key='tab2_unemp')
        store_type = st.selectbox("Select Type", options=type_options, key='tab2_type')
        store_size = st.selectbox("Select Size of the store", options=size_options, key='tab2_size')
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2023, step=1, key='tab2_year')

        # Lagged Sales and Holiday Inputs
        lagged_sales_1_weeks = st.number_input("Lagged Sales (1 week ago)", value=0.0, step=10.0, key='tab2_lag1')
        lagged_sales_2_weeks = st.number_input("Lagged Sales (2 weeks ago)", value=0.0, step=10.0, key='tab2_lag2')
        lagged_sales_4_weeks = st.number_input("Lagged Sales (4 weeks ago)", value=0.0, step=10.0, key='tab2_lag4')
        lagged_sales_12_weeks = st.number_input("Lagged Sales (12 weeks ago)", value=0.0, step=10.0, key='tab2_lag12')
        is_holiday_week = st.checkbox("Is Holiday Week?", value=False, key='tab2_holidayweek')
        lagged_holiday_1_week = st.checkbox("Lagged Holiday (1 week ago)?", value=False, key='tab2_hol1')
        lagged_holiday_2_weeks = st.checkbox("Lagged Holiday (2 weeks ago)?", value=False, key='tab2_hol2')

        # Predict button and prediction
        if st.button("Predict Weekly Sales", key='tab2'):
            input_data = np.array([[store, dept, int(is_holiday), temperature, fuel_price, cpi, unemployment, 
                                    store_type, store_size, lagged_sales_1_weeks, lagged_sales_2_weeks, 
                                    lagged_sales_4_weeks, lagged_sales_12_weeks, 
                                    int(is_holiday_week), int(lagged_holiday_1_week), 
                                    int(lagged_holiday_2_weeks), year]])
            prediction_no_markdown = loaded_model_no_markdown.predict(input_data)
            st.session_state['prediction_no_markdown'] = prediction_no_markdown
            st.session_state['markdowns'] = 'No'
            st.success(f"Predicted Weekly Sales: ${prediction_no_markdown[0]:,.2f}")

elif selected == 'Impact Analysis':
    # Ensure that prediction values are available
    if 'prediction' in st.session_state and 'prediction_no_markdown' in st.session_state:
        prediction = st.session_state['prediction']
        prediction_no_markdown = st.session_state['prediction_no_markdown']

        # Calculate markdown effect (assuming they are arrays)
        markdown_effect = prediction - prediction_no_markdown
        
        # Display the markdown effect summary
        st.subheader("Markdown Effect on Sales:")
        
        if isinstance(markdown_effect, (np.ndarray, list)):
            # Summarize the markdown effect
            average_effect = np.mean(markdown_effect)
            st.text(f"Average Markdown Effect on Sales: ${average_effect:,.2f}")
        else:
            # If markdown_effect is a single value
            st.success(f"Markdown Effect: ${markdown_effect:,.2f}")

        # Plot the markdown effect
        st.subheader("Markdown Effect Visualization")
        
        plt.figure()
        if isinstance(markdown_effect, (np.ndarray, list)):
            plt.plot(markdown_effect, label='Markdown Effect', color='blue', marker='o')
            plt.axhline(0, color='black', linewidth=0.5)
        else:
            plt.bar(['Markdown Effect'], [markdown_effect], color='blue')

        plt.ylabel('Sales Difference')
        plt.title('Difference in Sales Due to Markdowns')
        plt.legend()
        st.pyplot(plt.gcf())

    else:
        st.warning("Please perform a sales prediction first.")

