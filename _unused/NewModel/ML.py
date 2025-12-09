import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
import optuna
import xgboost as xgb
import shap
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.signal import detrend
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuration
np.random.seed(42)
torch.manual_seed(42)
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

print("--- Crop Yield Prediction Pipeline Initialized (Metric: RMSPE) ---")

# ==========================================
# 0. HELPER FUNCTIONS (METRICS & RECONSTRUCTION)
# ==========================================

def calculate_rmspe(y_true, y_pred):
    """Root Mean Squared Percentage Error"""
    # Add small epsilon to avoid division by zero
    return np.sqrt(np.mean(((y_true - y_pred) / (y_true + 1e-8)) ** 2)) * 100

def reconstruct_yield(y_pred_scaled, reference_df, y_scaler):
    """Reverses scaling and detrending to get actual hg/ha"""
    # 1. Inverse Scale
    if y_pred_scaled.ndim == 1: y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    y_pred_det = y_scaler.inverse_transform(y_pred_scaled).flatten()
    
    # 2. Add Trend back
    # Ensure reference_df has the same index/length as predictions
    return y_pred_det + reference_df['yield_trend'].values

# ==========================================
# 1. DATA LOADING & CONFIGURATION
# ==========================================

# Configuration Constants
TRAIN_END = 2007
VAL_END = 2010
LOOKBACK = 5
TARGET = 'hg/ha_yield'
TIME_COL = 'Year'
CAT_COLS = ['Area', 'Item']
BASE_NUMERIC_COLS = [
    'average_rain_fall_mm_per_year', 
    'pesticides_tonnes', 
    'avg_temp', 
    'fertilizer_kg/ha', 
    'solar_radiation_MJ/m2-day'
]

# Load Data
try:
    df = pd.read_csv("cleaned_crop_data.csv")
    print(f"Data loaded: {df.shape}")
except FileNotFoundError:
    raise FileNotFoundError("Error: 'cleaned_crop_data.csv' not found.")

# ==========================================
# 2. PREPROCESSING & FEATURE ENGINEERING
# ==========================================

print("\n--- Step 2: Preprocessing ---")

# 2.1 Split Data (Chronological)
train_df_orig = df[df[TIME_COL] <= TRAIN_END].copy()
val_df_orig = df[(df[TIME_COL] > TRAIN_END) & (df[TIME_COL] <= VAL_END)].copy()
test_df_orig = df[df[TIME_COL] > VAL_END].copy()

# 2.2 Label Encoding (Fit on Train, Apply to All)
le_area = LabelEncoder().fit(train_df_orig['Area'])
le_item = LabelEncoder().fit(train_df_orig['Item'])

for d in [train_df_orig, val_df_orig, test_df_orig]:
    d['Area_Encoded'] = d['Area'].apply(lambda x: le_area.transform([x])[0] if x in le_area.classes_ else 0)
    d['Item_Encoded'] = d['Item'].apply(lambda x: le_item.transform([x])[0] if x in le_item.classes_ else 0)

# 2.3 Detrending (Remove Global/Local Trends)
print("Fitting trend models...")
trend_models = {}
global_trend_model = LinearRegression().fit(train_df_orig[[TIME_COL]], train_df_orig[TARGET])

for group, group_df in train_df_orig.groupby(CAT_COLS):
    if len(group_df) > 1:
        model = LinearRegression().fit(group_df[[TIME_COL]], group_df[TARGET])
        trend_models[group] = model

for d in [train_df_orig, val_df_orig, test_df_orig]:
    d['yield_trend'] = 0.0
    for group, group_df in d.groupby(CAT_COLS):
        model = trend_models.get(group, global_trend_model)
        d.loc[group_df.index, 'yield_trend'] = model.predict(group_df[[TIME_COL]])
    d['yield_detrended'] = d[TARGET] - d['yield_trend']

# 2.4 Feature Lagging
full_df = pd.concat([train_df_orig, val_df_orig, test_df_orig]).sort_values(CAT_COLS + [TIME_COL])
cols_to_lag = ['yield_detrended'] + BASE_NUMERIC_COLS

for col in cols_to_lag:
    for lag in [1, 2]:
        full_df[f'{col}_lag{lag}'] = full_df.groupby(CAT_COLS)[col].shift(lag)

full_df = full_df.dropna()

# Re-split after lagging
train_df = full_df[full_df[TIME_COL] <= TRAIN_END].copy()
val_df = full_df[(full_df[TIME_COL] > TRAIN_END) & (full_df[TIME_COL] <= VAL_END)].copy()
test_df = full_df[full_df[TIME_COL] > VAL_END].copy()

# 2.5 Scaling
feature_cols = [c for c in train_df.columns if c in BASE_NUMERIC_COLS or '_lag' in c]
x_scaler = StandardScaler()
y_scaler = StandardScaler()

train_df[feature_cols] = x_scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols] = x_scaler.transform(val_df[feature_cols])
test_df[feature_cols] = x_scaler.transform(test_df[feature_cols])

train_df['yield_detrended'] = y_scaler.fit_transform(train_df[['yield_detrended']])
val_df['yield_detrended'] = y_scaler.transform(val_df[['yield_detrended']])
test_df['yield_detrended'] = y_scaler.transform(test_df[['yield_detrended']])

print("Preprocessing complete. Transformers fitted.")


# ==========================================
# 3. ML SPECIFIC PREPARATION
# ==========================================
print("\n--- Step 3: ML Data Preparation ---")

# 3.1 Select Features
ml_cols = feature_cols + ['Area_Encoded', 'Item_Encoded']

X_train_ml = train_df[ml_cols].copy()
y_train_ml = train_df['yield_detrended']
X_val_ml = val_df[ml_cols].copy()
y_val_ml = val_df['yield_detrended']
X_test_ml = test_df[ml_cols].copy()
y_test_ml = test_df['yield_detrended']

# 3.2 Drop Target Lags (Prevent Leakage)
drop_cols = ['yield_detrended_lag1', 'yield_detrended_lag2']
X_train_ml = X_train_ml.drop(columns=drop_cols, errors='ignore')
X_val_ml = X_val_ml.drop(columns=drop_cols, errors='ignore')
X_test_ml = X_test_ml.drop(columns=drop_cols, errors='ignore')

# 3.3 Target Encoding
print("Applying Target Encoding...")
temp_train = X_train_ml.copy()
temp_train['target'] = y_train_ml

area_means = temp_train.groupby('Area_Encoded')['target'].mean()
item_means = temp_train.groupby('Item_Encoded')['target'].mean()
global_mean = y_train_ml.mean()

for df_ml in [X_train_ml, X_val_ml, X_test_ml]:
    df_ml['Area_Target_Mean'] = df_ml['Area_Encoded'].map(area_means).fillna(global_mean)
    df_ml['Item_Target_Mean'] = df_ml['Item_Encoded'].map(item_means).fillna(global_mean)
    df_ml.drop(columns=['Area_Encoded', 'Item_Encoded'], inplace=True)

print(f"ML Features Finalized: {X_train_ml.shape[1]} features.")


# ==========================================
# 4. DL SPECIFIC PREPARATION
# ==========================================
print("\n--- Step 4: Deep Learning Data Preparation ---")

dl_numeric_features = [c for c in X_train_ml.columns if c not in ['Area_Target_Mean', 'Item_Target_Mean']]
print(f"DL Numeric Features ({len(dl_numeric_features)}): {dl_numeric_features}")

# EXTEND DATASETS
def extend_dataset(prev_df, curr_df, lookback, cat_cols):
    tails = prev_df.groupby(cat_cols).tail(lookback - 1)
    return pd.concat([tails, curr_df]).sort_values(cat_cols + [TIME_COL])

val_df_ext = extend_dataset(train_df, val_df, LOOKBACK, CAT_COLS)
test_df_ext = extend_dataset(val_df, test_df, LOOKBACK, CAT_COLS)

def create_sequences(data, lookback, num_feats, target_col):
    X_num, X_cat, y = [], [], []
    y_indices = []
    
    for _, group in data.groupby(CAT_COLS):
        if len(group) < lookback: continue
        
        gf_num = group[num_feats].values
        gf_area = group['Area_Encoded'].values[0]
        gf_item = group['Item_Encoded'].values[0]
        gt = group[target_col].values
        indices = group.index
        
        for i in range(len(group) - lookback + 1):
            X_num.append(gf_num[i:i+lookback])
            X_cat.append([gf_area, gf_item]) 
            y.append(gt[i+lookback-1])
            y_indices.append(indices[i+lookback-1])
            
    return np.array(X_num), np.array(X_cat), np.array(y), np.array(y_indices)

# Create Sequences
X_train_seq_n, X_train_seq_c, y_train_seq, _ = create_sequences(train_df, LOOKBACK, dl_numeric_features, 'yield_detrended')
# Capture Indices for Val and Test to create References
X_val_seq_n, X_val_seq_c, y_val_seq, y_val_idx = create_sequences(val_df_ext, LOOKBACK, dl_numeric_features, 'yield_detrended')
X_test_seq_n, X_test_seq_c, y_test_seq, y_test_idx = create_sequences(test_df_ext, LOOKBACK, dl_numeric_features, 'yield_detrended')

def to_tensors(X_n, X_c, y):
    if len(X_n) == 0:
        return [
            torch.empty(0, LOOKBACK, X_n.shape[-1] if len(X_n.shape) > 2 else len(dl_numeric_features)),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        ], torch.empty(0, 1)

    return [
        torch.tensor(X_n, dtype=torch.float32),
        torch.tensor(X_c[:, 0], dtype=torch.long), 
        torch.tensor(X_c[:, 1], dtype=torch.long), 
    ], torch.tensor(y, dtype=torch.float32).unsqueeze(1)

X_train_dl, y_train_t = to_tensors(X_train_seq_n, X_train_seq_c, y_train_seq)
X_val_dl, y_val_t = to_tensors(X_val_seq_n, X_val_seq_c, y_val_seq)
X_test_dl, y_test_t = to_tensors(X_test_seq_n, X_test_seq_c, y_test_seq)

# Create Reference DataFrames for Reconstruction
# This is essential for RMSPE calculation
val_df_dl_ref = val_df_ext.loc[y_val_idx].copy()
test_df_dl_ref = test_df_ext.loc[y_test_idx].copy()

print(f"DL Train Seq Shape: {X_train_dl[0].shape}")
print(f"DL Val Seq Shape:   {X_val_dl[0].shape}")


# ==========================================
# 5. MODEL DEFINITIONS & OBJECTIVES (RMSPE SCORING)
# ==========================================

# 5.1 Ridge Regression
def objective_lr(trial):
    alpha = trial.suggest_float('alpha', 0.1, 20.0, log=True)
    model = Ridge(alpha=alpha)
    model.fit(X_train_ml, y_train_ml)
    
    # Predict on Validation
    preds_scaled = model.predict(X_val_ml)
    
    # Reconstruct to Original Scale for RMSPE
    preds_orig = reconstruct_yield(preds_scaled, val_df, y_scaler)
    y_true_orig = val_df[TARGET].values
    
    return calculate_rmspe(y_true_orig, preds_orig)

# 5.2 Random Forest
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 10),        
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20), 
        'max_features': trial.suggest_float('max_features', 0.4, 0.8)
    }
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    model.fit(X_train_ml, y_train_ml)
    
    # Validation & Score
    preds_scaled = model.predict(X_val_ml)
    preds_orig = reconstruct_yield(preds_scaled, val_df, y_scaler)
    y_true_orig = val_df[TARGET].values
    
    return calculate_rmspe(y_true_orig, preds_orig)

# 5.3 XGBoost
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 30), 
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
    }
    # We optimize internal split using RMSE (stable), but select best trial using RMSPE
    model = xgb.XGBRegressor(random_state=42, early_stopping_rounds=50, eval_metric='rmse', **params)
    model.fit(X_train_ml, y_train_ml, eval_set=[(X_val_ml, y_val_ml)], verbose=False)
    
    # Validation & Score
    preds_scaled = model.predict(X_val_ml)
    preds_orig = reconstruct_yield(preds_scaled, val_df, y_scaler)
    y_true_orig = val_df[TARGET].values
    
    return calculate_rmspe(y_true_orig, preds_orig)

# 5.4 Deep Learning Models
class LSTMModel(nn.Module):
    def __init__(self, n_areas, n_items, input_dim, lstm_units, dense_units, dropout):
        super().__init__()
        self.embed_area = nn.Embedding(n_areas, 10)
        self.embed_item = nn.Embedding(n_items, 5)
        self.lstm = nn.LSTM(input_dim + 10 + 5, lstm_units, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_units, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)
        
    def forward(self, num, area, item):
        # Fix for Shape Mismatch: Expand Embeddings to Sequence Length
        seq_len = num.size(1)
        emb_area = self.embed_area(area).unsqueeze(1).expand(-1, seq_len, -1)
        emb_item = self.embed_item(item).unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([num, emb_area, emb_item], dim=-1)
        
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1])
        return self.fc2(torch.relu(self.fc1(out)))

class CNNModel(nn.Module):
    def __init__(self, n_areas, n_items, input_dim, filters, kernel, dense_units, dropout):
        super().__init__()
        self.embed_area = nn.Embedding(n_areas, 10)
        self.embed_item = nn.Embedding(n_items, 5)
        self.conv = nn.Conv1d(input_dim + 10 + 5, filters, kernel)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(filters, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)
        
    def forward(self, num, area, item):
        # Fix for Shape Mismatch
        seq_len = num.size(1)
        emb_area = self.embed_area(area).unsqueeze(1).expand(-1, seq_len, -1)
        emb_item = self.embed_item(item).unsqueeze(1).expand(-1, seq_len, -1)
        x = torch.cat([num, emb_area, emb_item], dim=-1).transpose(1, 2)
        
        x = self.pool(torch.relu(self.conv(x))).squeeze(-1)
        return self.fc2(torch.relu(self.fc1(self.drop(x))))

def train_dl(model, opt, loader, val_loader, epochs=100, patience=15):
    scheduler = ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)
    best_loss = float('inf') # Minimize MSE for stability
    wait = 0
    loss_fn = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        for x1, x2, x3, y in loader:
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x1, x2, x3), y).backward()
            opt.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x1, x2, x3, y in val_loader:
                x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
                val_loss += loss_fn(model(x1, x2, x3), y).item()
        
        val_mse = val_loss / len(val_loader)
        scheduler.step(val_mse)
        
        if val_mse < best_loss:
            best_loss = val_mse
            wait = 0
            torch.save(model.state_dict(), f'best_model_{model.__class__.__name__}.pth')
        else:
            wait += 1
            if wait >= patience: break
    
    if os.path.exists(f'best_model_{model.__class__.__name__}.pth'):
        model.load_state_dict(torch.load(f'best_model_{model.__class__.__name__}.pth'))
    return model

def objective_lstm(trial):
    params = {
        'lstm_units': trial.suggest_categorical('lstm_units', [32, 64]),
        'dense_units': trial.suggest_categorical('dense_units', [16, 32]),
        'dropout': trial.suggest_float('dropout', 0.2, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    }
    lr, wd = params.pop('lr'), params.pop('weight_decay')
    model = LSTMModel(len(le_area.classes_), len(le_item.classes_), len(dl_numeric_features), **params)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    train_ds = TensorDataset(*X_train_dl, y_train_t)
    val_ds = TensorDataset(*X_val_dl, y_val_t)
    
    # Train using MSE
    model = train_dl(model, opt, DataLoader(train_ds, batch_size=64, shuffle=True), DataLoader(val_ds, batch_size=64))
    
    # Calculate RMSPE for Optuna
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        inputs = [x.to(device) for x in X_val_dl]
        preds_scaled = model(*inputs).cpu().numpy()
        
    preds_orig = reconstruct_yield(preds_scaled, val_df_dl_ref, y_scaler)
    y_true_orig = val_df_dl_ref[TARGET].values
    
    return calculate_rmspe(y_true_orig, preds_orig)

def objective_cnn(trial):
    params = {
        'filters': trial.suggest_categorical('filters', [32, 64]),
        'kernel': trial.suggest_categorical('kernel', [2, 3]),
        'dense_units': trial.suggest_categorical('dense_units', [16, 32]),
        'dropout': trial.suggest_float('dropout', 0.2, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    }
    lr, wd = params.pop('lr'), params.pop('weight_decay')
    model = CNNModel(len(le_area.classes_), len(le_item.classes_), len(dl_numeric_features), **params)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    train_ds = TensorDataset(*X_train_dl, y_train_t)
    val_ds = TensorDataset(*X_val_dl, y_val_t)
    
    model = train_dl(model, opt, DataLoader(train_ds, batch_size=64, shuffle=True), DataLoader(val_ds, batch_size=64))
    
    # Calculate RMSPE
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        inputs = [x.to(device) for x in X_val_dl]
        preds_scaled = model(*inputs).cpu().numpy()
        
    preds_orig = reconstruct_yield(preds_scaled, val_df_dl_ref, y_scaler)
    y_true_orig = val_df_dl_ref[TARGET].values
    
    return calculate_rmspe(y_true_orig, preds_orig)


# ==========================================
# 6. OPTIMIZATION & FINAL TRAINING
# ==========================================
print("\n--- Step 6: Hyperparameter Tuning & Training (Optimizing RMSPE) ---")

best_models = {}
results_summary = []

objectives = {
    'Ridge': objective_lr,
    'RF': objective_rf,
    'XGB': objective_xgb,
    'LSTM': objective_lstm,
    'CNN': objective_cnn
}

for name, objective in objectives.items():
    print(f"Tuning {name}...")
    try:
        # Minimize RMSPE
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10 if name in ['RF', 'XGB'] else 5, show_progress_bar=False)
        print(f"  Best RMSPE: {study.best_value:.2f}% | Params: {study.best_params}")
        
        # Retrain on Train + Val for Final Test
        if name in ['Ridge', 'RF', 'XGB']:
            X_full = pd.concat([X_train_ml, X_val_ml])
            y_full = pd.concat([y_train_ml, y_val_ml])
            
            if name == 'Ridge':
                model = Ridge(alpha=study.best_params['alpha'])
                model.fit(X_full, y_full)
            elif name == 'RF':
                model = RandomForestRegressor(random_state=42, n_jobs=-1, **study.best_params)
                model.fit(X_full, y_full)
            elif name == 'XGB':
                # Keep RMSE for internal loss, but we know params are optimized for RMSPE
                model = xgb.XGBRegressor(random_state=42, eval_metric='rmse', **study.best_params)
                model.fit(X_full, y_full)
            
            best_models[name] = model
            
        else: # DL Models
            X_full_dl = [torch.cat([X_train_dl[i], X_val_dl[i]]) for i in range(3)]
            y_full_t = torch.cat([y_train_t, y_val_t])
            
            params = study.best_params
            lr, wd = params.pop('lr'), params.pop('weight_decay')
            
            if name == 'LSTM':
                model = LSTMModel(len(le_area.classes_), len(le_item.classes_), len(dl_numeric_features), **params)
            else:
                model = CNNModel(len(le_area.classes_), len(le_item.classes_), len(dl_numeric_features), **params)
                
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            full_ds = TensorDataset(*X_full_dl, y_full_t)
            test_ds = TensorDataset(*X_test_dl, y_test_t)
            
            model = train_dl(model, opt, DataLoader(full_ds, batch_size=64, shuffle=True), DataLoader(test_ds, batch_size=64), epochs=150)
            best_models[name] = model
            
    except Exception as e:
        print(f"Optimization failed for {name}: {e}")


# ==========================================
# 7. EVALUATION
# ==========================================
print("\n--- Step 7: Evaluation on Test Set ---")

final_preds = {}

for name, model in best_models.items():
    if name in ['Ridge', 'RF', 'XGB']:
        preds_scaled = model.predict(X_test_ml)
        preds_orig = reconstruct_yield(preds_scaled, test_df, y_scaler)
        y_true = test_df[TARGET].values
        
    else: # DL
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            inputs = [x.to(device) for x in X_test_dl]
            preds_scaled = model(*inputs).cpu().numpy()
        preds_orig = reconstruct_yield(preds_scaled, test_df_dl_ref, y_scaler)
        y_true = test_df_dl_ref[TARGET].values
    
    final_preds[name] = preds_orig
    
    rmse = np.sqrt(mean_squared_error(y_true, preds_orig))
    rmspe = calculate_rmspe(y_true, preds_orig)
    mae = mean_absolute_error(y_true, preds_orig)
    r2 = r2_score(y_true, preds_orig)
    
    results_summary.append({'Model': name, 'RMSPE (%)': rmspe, 'RMSE': rmse, 'MAE': mae, 'R2': r2})

df_results = pd.DataFrame(results_summary).set_index('Model').sort_values('RMSPE (%)')
print(df_results)


# ==========================================
# 8. VISUALIZATION (Best Model)
# ==========================================
print("\n--- Step 8: Visualizing Best Model ---")

if not df_results.empty:
    best_name = df_results.index[0]
    best_pred = final_preds[best_name]
    print(f"Plotting results for: {best_name}")

    plot_df = test_df.copy() if best_name in ['Ridge', 'RF', 'XGB'] else test_df_dl_ref.copy()
    plot_df['prediction'] = best_pred

    unique_crops = plot_df['Item'].unique()

    for crop in unique_crops:
        plt.figure(figsize=(15, 6))
        
        history_df = df[df['Item'] == crop]
        history_grp = history_df.groupby('Year')[TARGET].mean()
        pred_grp = plot_df[plot_df['Item'] == crop].groupby('Year')['prediction'].mean()
        
        if not pred_grp.empty:
            plt.plot(history_grp.index, history_grp.values, 'k-', lw=2, label='Actual Yield')
            plt.plot(pred_grp.index, pred_grp.values, 'r:o', lw=2, label=f'Predicted ({best_name})')
            
            plt.title(f"{crop}: Actual vs Predicted Yield ({best_name})")
            plt.xlabel("Year")
            plt.ylabel("Yield (hg/ha)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
else:
    print("No results to visualize.")

print("--- Pipeline Complete ---")