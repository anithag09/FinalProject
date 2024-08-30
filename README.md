# Retail Sales Prediction

This project aims to predict weekly sales for different departments across various stores using a combination of machine learning models. The project focuses on understanding the impact of various factors, including markdowns, economic indicators, and seasonal trends, on sales.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Model Evaluation](#model-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Project Overview
This project involves building machine learning models to predict weekly sales across different stores and departments. The goal is to provide accurate sales forecasts to assist in marketing strategies, and resource allocation.

## Dataset
The dataset consists of the following files:
- `sales.csv`: Contains historical sales data.
- `features.csv`: Includes store-level information such as temperature, fuel price, and markdowns.
- `stores.csv`: Provides store-level metadata like store size.

The dataset spans several years and includes key variables such as `Weekly_Sales`, `Temperature`, `Fuel_Price`, `MarkDown1-5`, `CPI`, and `Unemployment`.

## Data Preprocessing
Data preprocessing steps include:
1. **Merging Datasets**: The `sales`, `features`, and `stores` datasets were merged on `Store`, `IsHoliday` and `Date` to create a unified dataframe.
2. **Handling Missing Values**: Missing values were imputed.

## Exploratory Data Analysis (EDA)
The EDA phase involved:
- Visualizing trends, seasonality, and the impact of holidays on sales.
- Analyzing correlations between sales and other features.
- Investigating the distribution of sales across different departments and stores.

## Feature Engineering
Key features created include:
- **Lagged Sales**: Sales data from previous weeks to capture temporal dependencies.
- **Rolling Averages**: Moving averages over 4, 12, and 26 weeks to smooth out trends.
- **Holiday Flags**: Indicators for holiday weeks and lagged holiday effects.

## Modeling
The project explored several models:
1. **Linear Regression**: A baseline model to establish a reference performance.
2. **Random Forest**: A robust tree-based model for capturing non-linear relationships.
3. **ARIMA**: A time-series model used to capture and forecast the temporal dependencies in sales data by considering both past values and errors.
4. **XGBoost Hybrid**: A powerful boosting algorithm that combines the strengths of multiple weak learners, optimized to handle non-linear patterns and interactions within the sales data.
5. **LSTM (Long Short-Term Memory)**: A deep learning model to capture complex temporal dependencies.

## Model Evaluation
The models were evaluated using the following metrics:
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

Special attention was given to performance during holiday weeks, as these periods are critical for sales forecasting. Simulated the sales with and withour markdowns.

## Installation
To run this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anithag09/FinalProject.git
   ```
2. **Install the Required Packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit App**:
    ```bash
    streamlit run RetailSalesPrediction.py
    ```
## Usage
To use the models for your own data:

- Prepare your dataset in the same format as the provided datasets.
- Run the preprocessing and feature engineering steps.
- Train the models and evaluate them using the provided scripts.

## Conclusion
The project successfully demonstrated the use of various machine learning models to predict weekly sales. The model XGBoost provided a strong performance by effectively capturing the residual components.
