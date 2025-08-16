# Advanced Time Series Forecasting for Climate Data using DeepAR & N-BEATS

As a data scientist focused on applying deep learning to complex predictive challenges, I developed this project to demonstrate a comprehensive workflow for multivariate time series forecasting. Using daily climate data for Delhi as a case study, this notebook showcases advanced techniques including data acquisition from Kaggle, robust preprocessing, feature engineering, and the implementation of state-of-the-art models like **DeepAR** and **N-BEATS** using the Darts library.

This work is designed to showcase my skills in implementing different models, explaining their mechanics, and benchmarking various methods in practice—rather than solely focusing on achieving the highest evaluation metrics. If a model shows simpler logic or lower accuracy, it is often intentional for the purpose of learning or illustrating fundamental concepts. The project highlights my ability to handle real-world data complexities and leverage powerful frameworks like PyTorch Lightning for efficient, GPU-accelerated training and evaluation.

-----

## Problem Statement and Goal of Project

Accurately forecasting climate variables like temperature is a critical task with applications in energy consumption planning, agriculture, and public health. This project tackles the challenge of predicting the mean daily temperature by leveraging not just historical temperature data, but also other meteorological factors (covariates).

The goals of this project are to:

1.  Build a robust pipeline for fetching, cleaning, and preparing multivariate time series data.
2.  Engineer both past-observed and future-known covariates to enrich the models' predictive power.
3.  Implement and train two powerful deep learning models: a probabilistic **DeepAR-style RNN (LSTM)** and the block-based **N-BEATS** architecture.
4.  Rigorously evaluate the models through backtesting on historical data to simulate real-world performance.
5.  Compare the models' accuracy using standard metrics (MAE and MAPE) to determine the most effective approach for this dataset.

-----

## Solution Approach

The solution is structured as a step-by-step process within the Jupyter Notebook, emphasizing best practices for time series analysis.

  - **Data Acquisition & Cleaning**: The "Daily Delhi Climate" dataset is downloaded directly from Kaggle Hub. The initial train and test files are merged, and the data is meticulously cleaned by removing duplicate dates and anomalous `meanpressure` readings.
  - **Feature Engineering (Covariates)**: To provide the models with richer context, two types of covariates are created:
      - **Past Covariates**: These are features whose historical values are known at the time of prediction. This includes `humidity`, `wind_speed`, `meanpressure`, and derived features like a 7-day moving average (`ma_7`), 7-day standard deviation (`std_7`), and the rate of change (`roc_1`) of the mean temperature.
      - **Future Covariates**: These are features whose future values can be known in advance. Calendar information such as the `day`, `month`, and `weekday` are one-hot encoded and used for this purpose.
  - **Data Preparation**: The target variable (`meantemp`) and covariates are converted into `TimeSeries` objects using the Darts library. The data is then split into training, validation, and test sets and scaled to the `[0, 1]` range to optimize model training.
  - **Model Implementation (Darts & PyTorch Lightning)**:
      - **DeepAR (RNN Model)**: An LSTM-based RNN is configured as a probabilistic forecasting model (similar to Amazon's DeepAR) that outputs a Gaussian likelihood distribution, allowing for uncertainty estimation in its predictions.
      - **N-BEATS Model**: The N-BEATS architecture, known for its strong performance without requiring extensive feature engineering, is implemented to forecast the target variable using past covariates.
  - **Training & Evaluation**: Both models are trained using a robust training loop managed by **PyTorch Lightning**, which includes callbacks for early stopping, model checkpointing, and learning rate monitoring to ensure efficiency and prevent overfitting. The models' performances are compared on a held-out test set.
  - **Backtesting**: To validate the models' real-world utility, **historical forecasting (backtesting)** is performed. This involves simulating predictions over multiple historical periods to get a more reliable estimate of the models' accuracy over time, evaluated with MAE, MAPE, SMAPE, and RMSE.

-----

## Technologies & Libraries

  - **Programming Language**: Python (via Jupyter Notebook)
  - **Core Time Series Library**: Darts
  - **Deep Learning Framework**: PyTorch, PyTorch Lightning
  - **Data Processing & Visualization**: Pandas, NumPy, Matplotlib, scikit-learn (`Scaler`)
  - **Data Retrieval**: Kaggle Hub API

-----

## Description about Dataset

The dataset is the **Daily Climate Time Series Data** for Delhi, India, sourced from Kaggle Hub. It covers the period from **January 1, 2013, to April 24, 2017**.

  - **Features**: `meantemp` (target), `humidity`, `wind_speed`, `meanpressure`.
  - **Data Size**: 1575 daily records after cleaning.
  - **Structure**: The data is a time-indexed Pandas DataFrame, which is subsequently split into training (70%), validation (15%), and test (15%) sets.

| date | meantemp | humidity | wind\_speed | meanpressure |
| :--- | :--- | :--- | :--- | :--- |
| 2013-01-01| 10.00 | 84.50 | 0.00 | 1015.67 |
| 2013-01-02| 7.40 | 92.00 | 2.98 | 1017.80 |
| 2013-01-03| 7.17 | 87.00 | 4.63 | 1018.67 |
| 2013-01-04| 8.67 | 71.33 | 1.23 | 1017.17 |
| 2013-01-05| 6.00 | 86.83 | 3.70 | 1016.50 |

-----

## Installation & Execution Guide

1.  **Prerequisites**:

      - Python 3.9+
      - Jupyter Notebook or JupyterLab
      - A Kaggle account with an API token (`kaggle.json`) set up for data download.
      - GPU hardware is recommended for faster training.

2.  **Install Dependencies**:

    ```bash
    pip install "darts[torch]" tensorflow numpy pandas matplotlib kagglehub scikit-learn
    ```

3.  **Execution**:

      - Open the `project_2.ipynb` notebook in your Jupyter environment.
      - Ensure your `kaggle.json` file is correctly placed to allow the `kagglehub` library to download the dataset.
      - Run the cells sequentially to perform data downloading, cleaning, feature engineering, model training, and evaluation.

-----

## Key Results / Performance

The models were trained and evaluated on the test set, with the following performance metrics (lower is better):

  - **DeepAR (RNN Model)**:
      - **MAE**: 0.0854
      - **MAPE**: 18.08%
  - **N-BEATS Model**:
      - **MAE**: 0.0573
      - **MAPE**: 13.64%

The **N-BEATS model demonstrated superior performance** on this task, achieving a lower error in its predictions. Additionally, the backtesting results for the DeepAR model provided a comprehensive performance overview across different time horizons:

  - **Backtesting MAE (DeepAR)**: 0.0851
  - **Backtesting MAPE (DeepAR)**: 16.39%

-----

## Screenshots / Sample Outputs

  - **Data Visualization**: The notebook includes plots of the raw mean temperature, showcasing the time series' trends and seasonality.
  - **Forecast Plot**: A final plot compares the actual temperature values from the test set against the probabilistic forecasts generated by the DeepAR model, including quantile bands to visualize prediction uncertainty. Another plot compares the point forecasts from both DeepAR and N-BEATS.

-----

## Additional Learnings / Reflections

This project provided valuable, hands-on experience in building a sophisticated deep learning forecasting pipeline.

  - **The Power of Covariates**: Engineering both past and future covariates demonstrated their significant impact on model performance. Using features like weather metrics and calendar data provides crucial context that autoregressive models can leverage.
  - **Probabilistic Forecasting**: Implementing a DeepAR-style model highlighted the value of probabilistic forecasts. For real-world applications, understanding the range of potential outcomes (uncertainty) is often more critical than a single point prediction.
  - **Framework Efficiency**: Using Darts with a PyTorch Lightning backend streamlined the entire experimental process. The high-level API of Darts simplifies complex tasks like data handling and backtesting, while Lightning's callbacks ensure robust and efficient model training. This combination is highly effective for rapid prototyping and rigorous model evaluation.

-----

## 👤 Author

## Mehran Asgari

**Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com).
**GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari).

-----

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.

> 💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*