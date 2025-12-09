# Machine Learning for Crop Yield Prediction

**Course:** EAEE4000 – Machine Learning Applications for Environmental Engineering (Fall 2025)
**Instructor:** Prof. Pierre Gentine, Columbia University

## 📌 Project Overview

This project uses different machine learning models to predict global crop yields (Rice, Wheat, Maize, and Barley) using environmental and agricultural data.

The main goal is to compare simple and advanced models and see which ones handle the complex relationships between weather, farming practices, and crop yields the best.

## 📂 Project Structure

The work is split into 5 main parts:

* **`Part_1_DataPrep.ipynb`**
  Clean and merge raw agricultural and weather data (NASA POWER), and handle missing values.

* **`Part_2_EDA.ipynb`**
  Explore the data with basic visualizations like trends, maps, and correlations.

* **`Part_3_PrepFeature.ipynb`**
  Create features such as lagged yields and seasonal (cyclic) weather variables.

* **`Part_4_PrepLabel.ipynb`**
  Finalize the target variables and align features and labels for model training.

* **`Part_5_...` (Modeling)**
  Train and test different models:

  * Baseline: Previous-year prediction
  * XGBoost
  * Feedforward Neural Network (PyTorch)
  * 1D-CNN for short-term patterns
  * LSTM for time-series forecasting

## 🛠️ Tech Stack

* Python 3.x
* Pandas, NumPy
* PyTorch, XGBoost, Scikit-learn
* Optuna (for hyperparameter tuning)
* Matplotlib, Seaborn, Plotly

## 🚀 How to Run

1. Install the required libraries (make sure `torch` and `xgboost` are installed).
2. Run the notebooks in order from Part 1 to Part 4 to generate the intermediate files.
3. Run any of the `Part_5` notebooks to train and evaluate the models.
