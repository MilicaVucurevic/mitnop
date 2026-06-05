# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:34:55 2026

@author: danij
"""

# %% dokumentacija
 
# Opis: Razvoj i treniranje LSTM modela na stratifikovanom skupu podataka.
#       Stratifikacija: 70% iz perioda finansijske krize + 70% iz covid perioda +
#                       70% iz normalnog perioda, spojeno hronoloski.
#       Arhitektura je identicna originalnom LSTM modelu:
#       dva LSTM sloja + Dropout + Dense izlazni sloj.
#       Cilj: ispitati da li ravnomerna zastupljenost sva tri tipa trzisnih
#       uslova u trening skupu poboljsava generalizaciju modela.
#
# Input:
#       - data/processed/X_train_mix.pkl, X_val_mix.pkl, X_test_mix.pkl
#       - data/processed/y_train_mix.pkl, y_val_mix.pkl, y_test_mix.pkl
#       - data/processed/dates_train_mix.pkl, dates_val_mix.pkl, dates_test_mix.pkl
#       - data/processed/scaler.pkl
#
# Output:
#       - data/processed/lstm_mix_model.keras
#       - data/processed/lstm_mix_history.pkl
#       - data/processed/lstm_mix_predictions.pkl
#       - data/processed/lstm_mix_metrics.pkl
 
# %% biblioteke
 
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
 
# %% ucitavanje podataka
 
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
 
X_train = load_pkl(' ../../data/processed/X_train_mix.pkl')
X_val   = load_pkl(' ../../data/processed/X_val_mix.pkl')
X_test  = load_pkl(' ../../data/processed/X_test_mix.pkl')
 
y_train = load_pkl(' ../../data/processed/y_train_mix.pkl')
y_val   = load_pkl(' ../../data/processed/y_val_mix.pkl')
y_test  = load_pkl(' ../../data/processed/y_test_mix.pkl')
 
dates_train = load_pkl(' ../../data/processed/dates_train_mix.pkl')
dates_val   = load_pkl(' ../../data/processed/dates_val_mix.pkl')
dates_test  = load_pkl(' ../../data/processed/dates_test_mix.pkl')
 
scaler = load_pkl(' ../../data/processed/scaler.pkl')
 
print("Uspesno ucitani podaci (pomesani/spojeni skup):")
print(f"  X_train : {X_train.shape}  | y_train : {y_train.shape}")
print(f"  X_val   : {X_val.shape}   | y_val   : {y_val.shape}")
print(f"  X_test  : {X_test.shape}   | y_test  : {y_test.shape}")
print(f"  Oblik (Samples, Timesteps, Features): {X_train.shape}")
 
# %% definisanje parametara modela
 
LSTM_UNITS_1   = 64
LSTM_UNITS_2   = 32
DROPOUT_RATE   = 0.10
DENSE_UNITS    = 16
LEARNING_RATE  = 0.001
BATCH_SIZE     = 32
MAX_EPOCHS     = 150
PATIENCE       = 15
N_FEATURES     = X_train.shape[2]
N_TIMESTEPS    = X_train.shape[1]
 
print("\nParametri modela:")
print(f"  Timesteps : {N_TIMESTEPS}")
print(f"  Features  : {N_FEATURES}")
print(f"  LSTM sloj 1: {LSTM_UNITS_1} neurona")
print(f"  LSTM sloj 2: {LSTM_UNITS_2} neurona")
print(f"  Dropout   : {DROPOUT_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epoha : {MAX_EPOCHS}")
print(f"  Patience  : {PATIENCE}")
 
# %% izgradnja arhitekture modela
 
tf.random.set_seed(42)
np.random.seed(42)
 
model = Sequential([
    LSTM(LSTM_UNITS_1,
         return_sequences=True,
         kernel_initializer='glorot_uniform',
         input_shape=(N_TIMESTEPS, N_FEATURES),
         name='lstm_1'),
    Dropout(DROPOUT_RATE, name='dropout_1'),
 
    LSTM(LSTM_UNITS_2,
         return_sequences=False,
         kernel_initializer='glorot_uniform',
         name='lstm_2'),
    Dropout(DROPOUT_RATE, name='dropout_2'),
 
    Dense(DENSE_UNITS, activation='relu', kernel_initializer='glorot_uniform', name='dense_1'),
    Dense(1, name='output')
])
 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mae']
)
 
model.summary()
 
# %% definisanje callback-ova
 
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)
 
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.6,
    patience=10,
    min_lr=1e-5,
    verbose=1
)
 
# %% treniranje modela
 
print("\nPokretanje treniranja LSTM modela (izmesani skup)...")
print("=" * 50)
 
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
 
epoha_zaustavljanja = early_stopping.stopped_epoch
if epoha_zaustavljanja > 0:
    best_epoch = epoha_zaustavljanja - PATIENCE + 1
    print(f"\nEarly Stopping aktiviran u epohi {epoha_zaustavljanja + 1}")
    print(f"Najbolje tezine su iz epohe: {best_epoch}")
else:
    print(f"\nTrening zavrsen bez Early Stopping-a ({MAX_EPOCHS} epoha)")
 
# %% grafik krive ucenja
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
axes[0].plot(history.history['loss'], label='Train Loss', color='steelblue')
axes[0].plot(history.history['val_loss'], label='Val Loss', color='red')
axes[0].set_title('Learning Curve - Loss (MSE)')
axes[0].set_xlabel('Epoha')
axes[0].set_ylabel('MSE')
axes[0].legend()
 
axes[1].plot(history.history['mae'], label='Train MAE', color='steelblue')
axes[1].plot(history.history['val_mae'], label='Val MAE', color='red')
axes[1].set_title('Learning Curve - MAE')
axes[1].set_xlabel('Epoha')
axes[1].set_ylabel('MAE')
axes[1].legend()
 
plt.suptitle('LSTM (izmesani skup) - Krive ucenja', fontsize=13)
plt.tight_layout()
#plt.savefig('../../data/processed/lstm_strat_learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()
 
# %% predikcije na validacionom i test skupu
 
y_val_pred_scaled  = model.predict(X_val)
y_test_pred_scaled = model.predict(X_test)
 
REGULAR_CONV_IDX = 6
 
def inverse_transform_target(y_scaled, scaler, col_idx, n_cols=9):
    dummy = np.zeros((len(y_scaled), n_cols))
    dummy[:, col_idx] = y_scaled.flatten()
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, col_idx]
 
y_val_true  = inverse_transform_target(y_val,             scaler, REGULAR_CONV_IDX)
y_val_pred  = inverse_transform_target(y_val_pred_scaled,  scaler, REGULAR_CONV_IDX)
y_test_true = inverse_transform_target(y_test,             scaler, REGULAR_CONV_IDX)
y_test_pred = inverse_transform_target(y_test_pred_scaled, scaler, REGULAR_CONV_IDX)
 
# %% racunanje metrika
 
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape
 
val_rmse,  val_mae,  val_mape  = compute_metrics(y_val_true,  y_val_pred)
test_rmse, test_mae, test_mape = compute_metrics(y_test_true, y_test_pred)
 
print("\nMetrike na VALIDACIONOM skupu:")
print(f"  RMSE : {val_rmse:.4f} USD/galon")
print(f"  MAE  : {val_mae:.4f} USD/galon")
print(f"  MAPE : {val_mape:.2f}%")
 
print("\nMetrike na TEST skupu:")
print(f"  RMSE : {test_rmse:.4f} USD/galon")
print(f"  MAE  : {test_mae:.4f} USD/galon")
print(f"  MAPE : {test_mape:.2f}%")
 
# %% vizualizacija predikcija
 
dates_val_dt  = pd.to_datetime(dates_val)
dates_test_dt = pd.to_datetime(dates_test)
 
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
 
axes[0].plot(dates_val_dt, y_val_true, color='steelblue', linewidth=1.5, label='Stvarne vrednosti')
axes[0].plot(dates_val_dt, y_val_pred, color='red', linewidth=1.5, linestyle='--', label='LSTM predikcije')
axes[0].fill_between(dates_val_dt, y_val_true, y_val_pred, alpha=0.15, color='red', label='Greška')
axes[0].set_title(f'Validacioni skup | RMSE={val_rmse:.4f} | MAE={val_mae:.4f} | MAPE={val_mape:.2f}%')
axes[0].set_ylabel('USD/galon')
axes[0].legend()
 
axes[1].plot(dates_test_dt, y_test_true, color='steelblue', linewidth=1.5, label='Stvarne vrednosti')
axes[1].plot(dates_test_dt, y_test_pred, color='red', linewidth=1.5, linestyle='--', label='LSTM predikcije')
axes[1].fill_between(dates_test_dt, y_test_true, y_test_pred, alpha=0.15, color='red', label='Greška')
axes[1].set_title(f'Test skup | RMSE={test_rmse:.4f} | MAE={test_mae:.4f} | MAPE={test_mape:.2f}%')
axes[1].set_ylabel('USD/galon')
axes[1].set_xlabel('Datum')
axes[1].legend()
 
plt.suptitle('LSTM (izmesani skup) - Predikcije vs Stvarne vrednosti', fontsize=13)
plt.tight_layout()
#plt.savefig('../../data/processed/lstm_strat_predikcije.png', dpi=150, bbox_inches='tight')
plt.show()
 
# %% residual plot
 
reziduali_test = y_test_true - y_test_pred
 
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
 
axes[0].plot(dates_test_dt, reziduali_test, color='darkorange', linewidth=0.8)
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0].set_title('Reziduali na test skupu (stvarna - predviđena vrednost)')
axes[0].set_ylabel('Greška (USD/galon)')
 
axes[1].hist(reziduali_test, bins=40, color='darkorange', alpha=0.7, edgecolor='white')
axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_title('Distribucija reziduala')
axes[1].set_xlabel('Greška (USD/galon)')
axes[1].set_ylabel('Frekvencija')
 
plt.tight_layout()
#plt.savefig('../../data/processed/lstm_strat_reziduali.png', dpi=150, bbox_inches='tight')
plt.show()
 
# %% cuvanje modela i rezultata
 
lstm_strat_metrics = {
    'model'     : 'LSTM_stratified',
    'val_rmse'  : val_rmse,
    'val_mae'   : val_mae,
    'val_mape'  : val_mape,
    'test_rmse' : test_rmse,
    'test_mae'  : test_mae,
    'test_mape' : test_mape,
}
 
lstm_strat_predictions = {
    'dates_test' : dates_test_dt,
    'y_true'     : y_test_true,
    'y_pred'     : y_test_pred,
    'dates_val'  : dates_val_dt,
    'val_true'   : y_val_true,
    'val_pred'   : y_val_pred,
}
 
os.makedirs('../../data/processed', exist_ok=True)
model.save('../../data/processed/lstm_strat_model.keras')
 
with open('../../data/processed/lstm_strat_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
 
with open('../../data/processed/lstm_strat_predictions.pkl', 'wb') as f:
    pickle.dump(lstm_strat_predictions, f)
 
with open('../../data/processed/lstm_strat_metrics.pkl', 'wb') as f:
    pickle.dump(lstm_strat_metrics, f)
 
print("\nSačuvano:")
print("  data/processed/lstm_strat_model.keras")
print("  data/processed/lstm_strat_history.pkl")
print("  data/processed/lstm_strat_predictions.pkl")
print("  data/processed/lstm_strat_metrics.pkl")
print(f"\n{'='*50}")
print("LSTM (STRATIFIKOVANI SKUP) TRENIRANJE ZAVRSENO")
print(f"{'='*50}")
print(f"  Val  RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | MAPE: {val_mape:.2f}%")
print(f"  Test RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | MAPE: {test_mape:.2f}%")

# %% zakljucak

# Metrike na VALIDACIONOM skupu:
#  RMSE : 0.1267 USD/galon
#  MAE  : 0.1037 USD/galon
#  MAPE : 4.03%

# Metrike na TEST skupu:
#  RMSE : 0.1377 USD/galon
#  MAE  : 0.1112 USD/galon
#  MAPE : 5.55%