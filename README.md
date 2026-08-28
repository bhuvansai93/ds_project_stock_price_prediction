# Do Machine Learning and Deep Learning Models Beat a Naive Baseline?

## A comparative study of next-day price prediction for five Indian stocks

This project investigates whether machine learning (ML) and deep
learning (DL) models produce a genuine improvement over a naive
persistence baseline when predicting the next trading day's closing
price of Indian equities.

Five Indian stocks drawn from five sectors are studied:

-   Reliance --- Energy
-   Infosys --- Information Technology
-   HDFC Bank --- Banking
-   Maruti --- Automobile
-   Sun Pharma --- Healthcare

The study focuses on next-day **price-level prediction** using
historical OHLCV data and examines whether increasingly complex models
actually improve forecasting performance over a simple persistence rule.

------------------------------------------------------------------------

## Research Questions

### RQ1

**Do machine learning and deep learning models produce a genuine
improvement over a naive persistence baseline when predicting next-day
closing prices of Indian equities?**

### RQ2

**Which learned model performs best, and is that ranking consistent
across the five stocks?**

RQ1 is the primary question because a comparison between learned models
is only meaningful after establishing whether any of them can beat a
rule requiring no learned model.

------------------------------------------------------------------------

## Project Objectives

1.  Build a leak-free next-day regression pipeline using raw OHLCV(Open, High, Low, Close, Volume) data
    for five Indian stocks from five sectors.
2.  Use a strictly chronological train-test split.
3.  Train and optimise a linear model, two tree ensembles and two
    recurrent architectures.
4.  Evaluate every learned model against a naive persistence baseline
    using multiple metrics.
5.  Explain performance differences through model coefficients,
    training/testing target-range shift, recurrent lookback length and
    random-seed sensitivity.
6.  Discuss the practical meaning and limitations of the findings.

------------------------------------------------------------------------

## Dataset

The project uses ten years of daily market data covering **2015--2024**,
obtained from Yahoo Finance.

The five selected stocks are:

  Sector                   Stock
  ------------------------ ------------
  Energy                   Reliance
  Information Technology   Infosys
  Banking                  HDFC Bank
  Automobile               Maruti
  Healthcare               Sun Pharma

Each dataset contains:

-   Open
-   High
-   Low
-   Close
-   Volume

The final modelling experiment uses raw OHLCV variables. Technical
indicators were investigated during the project but are not part of the
final modelling setup.

------------------------------------------------------------------------

## Target Variable

The target is the next trading day's closing price:

``` text
Target(t) = Close(t + 1)
```

The target is created by shifting the closing-price series one trading
day forward.

The project uses the **price level** rather than return or direction
because the main research question tests whether learned models can beat
a naive persistence rule defined on the price level.

------------------------------------------------------------------------

## Data Preparation

The preprocessing workflow includes:

-   Cleaning the downloaded stock files
-   Removing metadata rows
-   Converting dates to datetime
-   Converting numerical columns to numeric types
-   Sorting observations chronologically
-   Removing duplicate dates
-   Setting the date as the index
-   Checking data quality and missing values
-   Creating the next-day target
-   Removing observations without a target
-   Saving the cleaned datasets to `cleaned_datasets/`

------------------------------------------------------------------------

## Train-Test Split

A chronological **75/25 split** is used:

``` text
First 75%  → Training data
Last 25%   → Testing data
```

The observations are not shuffled. This preserves temporal ordering and
prevents future observations from being used during model training.

Feature scaling is fitted using training data and then applied to the
test data.

------------------------------------------------------------------------

# Models

The primary comparison contains nine learned ML/DL models.

### Linear Regression

A linear regression model is used as the main traditional
machine-learning approach.

### Random Forest Regression

A nonlinear tree-based ensemble model is evaluated in both baseline and
tuned forms.

### XGBoost Regression

A gradient-boosted tree model is evaluated in both baseline and tuned
forms.

### LSTM

A recurrent neural network designed for sequential data. Both baseline
and tuned LSTM models are evaluated.

### Bidirectional LSTM

A bidirectional recurrent architecture is evaluated in both baseline and
tuned forms.

Therefore, the nine learned models are:

1.  Linear Regression
2.  Random Forest
3.  Tuned Random Forest
4.  XGBoost
5.  Tuned XGBoost
6.  LSTM
7.  Tuned LSTM
8.  BiLSTM
9.  Tuned BiLSTM

------------------------------------------------------------------------

## Hyperparameter Optimisation

Random Forest and XGBoost are tuned using:

-   RandomizedSearchCV
-   TimeSeriesSplit
-   5 cross-validation splits
-   50 sampled configurations

The recurrent models are tuned using Keras Tuner RandomSearch.

The deep-learning search considers parameters such as:

-   LSTM units
-   Dropout rates
-   Dense-layer units
-   Learning rate

The test set is kept separate from hyperparameter selection.

------------------------------------------------------------------------

# Naive Persistence Baseline

The naive persistence method is **not treated as an ML/DL model**. It is
a benchmark used to determine whether learned models provide genuine
improvement.

It assumes:

``` text
Predicted Close(t + 1) = Close(t)
```

In other words, tomorrow's closing price is predicted to be today's
closing price.

This benchmark is central to the project because daily stock-price
levels are highly persistent.

------------------------------------------------------------------------

# Evaluation Metrics

The models are evaluated using:

-   **MAE** --- Mean Absolute Error
-   **MSE** --- Mean Squared Error
-   **RMSE** --- Root Mean Squared Error
-   **MAPE** --- Mean Absolute Percentage Error
-   **R²** --- Coefficient of Determination

For MAE, MSE, RMSE and MAPE, lower values indicate better performance.
For R², higher values indicate better fit, but R² is interpreted
together with the naive benchmark.

------------------------------------------------------------------------

# Main Results

Linear Regression was the best-performing **learned ML/DL model based on
RMSE for all five stocks**:

  Stock        Best Learned Model
  ------------ --------------------
  Reliance     Linear Regression
  Infosys      Linear Regression
  HDFC Bank    Linear Regression
  Maruti       Linear Regression
  Sun Pharma   Linear Regression

However, Linear Regression did **not** outperform the naive persistence
baseline on any of the five stocks.

Therefore, the important distinction is:

``` text
Best learned model ≠ Best overall forecasting benchmark
```

------------------------------------------------------------------------

# Main Finding: Price Persistence

The learned models produced high R² values, but the naive persistence
baseline also produced very high performance.

Further analysis showed that Linear Regression predictions were almost
identical to the naive predictions. The current closing price had the
dominant fitted coefficient, while the other OHLCV coefficients were
close to zero.

This indicates that Linear Regression largely reproduced the persistence
relationship between today's and tomorrow's price.

Therefore:

> A high R² in next-day price-level prediction does not necessarily
> demonstrate useful forecasting skill.

------------------------------------------------------------------------

# Why Random Forest and XGBoost Were Weaker

The project compares the actual target-price ranges in the training and
testing periods.

Many test observations were above the maximum target price observed
during training:

  Stock          Test Values Above Training Maximum   Outside Training Range
  ------------ ------------------------------------ ------------------------
  Reliance                                      223                   36.14%
  Infosys                                       100                   16.21%
  HDFC Bank                                     173                   28.04%
  Maruti                                        390                   63.21%
  Sun Pharma                                    351                   56.89%

This shows a substantial upward shift in the target-price distribution.

The shift is treated as a **contributing explanation**, rather than
proof of a single cause, for weaker tree-model generalisation. Tree
ensembles are primarily interpolation-oriented and have limited ability
to smoothly extrapolate beyond patterns represented in training data.

------------------------------------------------------------------------

# Linear Regression Error Analysis

Because Linear Regression was the strongest learned model for all five
stocks, additional error analysis was performed.

The analysis includes:

-   Actual vs predicted prices
-   Absolute error over time
-   Largest prediction errors
-   Error during large price movements
-   Error during normal price movements

Large price movements are defined as the top 10% of absolute daily
percentage movements.

The analysis shows that prediction errors increase substantially during
large price movements.

  ----------------------------------------------------------------------------
  Stock                  MAE        Maximum   Large-Movement   Normal-Movement
                             Absolute Error            Error             Error
  ----------- -------------- -------------- ---------------- -----------------
  Reliance             11.96         109.92            36.80              9.18

  Infosys              15.06         118.12            47.63             11.41

  HDFC Bank             6.65          67.65            21.54              4.98

  Maruti               94.48         802.14           302.83             71.17

  Sun Pharma           10.89          59.72            32.34              8.49
  ----------------------------------------------------------------------------

------------------------------------------------------------------------

# Recurrent Lookback-Length Experiment

A supplementary experiment investigates whether providing LSTM and
BiLSTM models with longer historical sequences improves performance.

The tested lookback windows are:

``` text
1, 2, 3, 5, 10, 15, 20, 30 days
```

Three random seeds are used for each configuration.

The results show that increasing the lookback length does not
consistently improve next-day price-level prediction. Longer windows
frequently produce higher RMSE than the short-window configurations.

This suggests that simply providing more historical observations does
not guarantee better recurrent-model performance.

------------------------------------------------------------------------

# Random Seed Sensitivity

A supplementary experiment also examines the effect of random seed on
recurrent-model results.

The results show that changing the random seed can produce substantial
changes in deep-learning RMSE.

In some cases, the variation caused by the seed is larger than the
difference between different deep-learning architectures.

Therefore, small performance differences between LSTM and BiLSTM should
not automatically be interpreted as evidence that one architecture is
definitively superior.

------------------------------------------------------------------------

# Overall Findings

### 1. Linear Regression was the best learned model

Linear Regression achieved the lowest RMSE among the learned ML/DL
models for all five stocks.

### 2. No learned model beat the naive benchmark

The naive persistence rule remained stronger than every learned model
for next-day price-level prediction.

### 3. Model complexity did not guarantee improvement

Random Forest, XGBoost, LSTM and BiLSTM did not consistently outperform
the simple linear approach.

### 4. High R² was strongly influenced by price persistence

The strong performance of both Linear Regression and the naive benchmark
indicates that current and next-day price levels are highly persistent.

### 5. Tree models struggled under target-range shift

The testing period contained many target values above the maximum target
value observed during training.

### 6. Large price movements were harder to predict

Prediction errors were substantially higher during the largest daily
price movements.

### 7. Longer recurrent windows did not consistently help

Increasing the lookback from 1 to 30 days did not produce a consistent
improvement.

------------------------------------------------------------------------

# Answer to the Research Questions

### RQ1

**Do machine learning and deep learning models produce a genuine
improvement over a naive persistence baseline?**

**Answer: No.**

None of the learned ML/DL models outperformed the naive persistence
baseline for the five selected stocks.

### RQ2

**Which learned model performs best, and is that ranking consistent
across the five stocks?**

**Answer: Linear Regression was the best learned model based on RMSE for
all five stocks.**

Thus, the best learned-model ranking was consistent across the selected
stocks.

Because only one stock represents each sector, the results should not be
interpreted as evidence that Linear Regression is necessarily the best
model for entire sectors.

------------------------------------------------------------------------

# Limitations

-   Only five Indian stocks were analysed.
-   One stock represents each selected sector, so sector-wide
    generalisation is limited.
-   The target is the price level rather than returns or direction.
-   The final modelling experiment uses only raw OHLCV variables.
-   The test period contains substantial target-price distribution
    shift.
-   Recurrent-model results show sensitivity to random seed.
-   Prediction accuracy was evaluated rather than actual trading
    profitability.
-   Transaction costs and portfolio-level performance were not
    evaluated.

------------------------------------------------------------------------

# Future Work

Future work could:

1.  Change the target from price level to daily return and formulate a
    directional classification problem.
2.  Investigate engineered features and technical indicators more
    systematically.
3.  Select stocks with substantially different volatility levels and
    test whether model rankings change.
4.  Use walk-forward validation to better represent repeated forecasting
    conditions.
------------------------------------------------------------------------

# Project Workflow

``` text
Data Collection
      ↓
Data Cleaning and Preparation
      ↓
Exploratory Data Analysis
      ↓
Create Next-Day Closing Price Target
      ↓
Chronological Train-Test Split
      ↓
Feature Scaling
      ↓
Linear Regression
      ↓
Random Forest + Tuning
      ↓
XGBoost + Tuning
      ↓
LSTM + Tuning
      ↓
BiLSTM + Tuning
      ↓
Naive Persistence Benchmark
      ↓
Model Comparison
      ↓
Target-Range Analysis
      ↓
Linear Regression Error Analysis
      ↓
Lookback-Length Experiment
      ↓
Seed-Sensitivity Analysis
      ↓
Conclusions
```

------------------------------------------------------------------------

# Repository Structure

``` text
ds_project_stock-price-prediction/
│
├── final_stock_code.ipynb
├── README.md
├── Datasets10/
│   ├── Reliance.csv
│   ├── Infosys.csv
│   ├── HDFC_Bank.csv
│   ├── Maruti.csv
│   └── Sun_Pharma.csv
│
├── cleaned_datasets/
│   ├── Reliance.csv
│   ├── Infosys.csv
│   ├── HDFC_Bank.csv
│   ├── Maruti.csv
│   └── Sun_Pharma.csv
│
├── requirements.txt
└── other project files
```

------------------------------------------------------------------------

# Technologies Used

-   Python
-   NumPy
-   Pandas
-   Matplotlib
-   Scikit-learn
-   SciPy
-   XGBoost
-   TensorFlow / Keras
-   Keras Tuner

The exact package versions used in the project are provided in
`requirements.txt`.

------------------------------------------------------------------------

# How to Run

1. Clone the repository.
2. Create and activate a Python environment.
3. Install the project dependencies:

```bash
pip install -r requirements.txt
```

4. Place the raw stock CSV files in the `Datasets10/` directory.
5. Open `final_stock_code.ipynb` in Jupyter Notebook or JupyterLab.
6. Run the notebook from the beginning.

The notebook performs data preparation, exploratory analysis, model
training, hyperparameter optimisation, evaluation, model comparison and
supplementary analysis.

During preprocessing, the cleaned datasets are saved in the
`cleaned_datasets/` directory.

------------------------------------------------------------------------

# Requirements

The project uses the following Python packages and versions:

```text
numpy==2.2.6
pandas==2.3.3
matplotlib==3.10.6
scikit-learn==1.7.2
scipy==1.16.2
xgboost==3.1.1
tensorflow==2.21.0
keras-tuner==1.4.8
```

These dependencies are also listed in `requirements.txt`.

------------------------------------------------------------------------

# Academic Context

This repository contains the implementation associated with an **MSc
Data Science final project at the University of Hertfordshire**.

The project is an academic investigation of next-day stock-price
prediction and does not constitute financial or investment advice.

------------------------------------------------------------------------

## Author

**Bhuvana Sai Puchakayala**

MSc Data Science\
University of Hertfordshire
