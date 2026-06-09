# -*- coding: utf-8 -*-
"""
Created on Sun May 31 2026

@author: Milica
"""

# %% dokumentacija

# Opis: Razvoj i treniranje GRU modela za predikciju cena regular_conv benzina.
#       GRU je multivarijavantna metoda - koristi cene goriva, sirove nafte i USD indeksa (2006-2021).
#       Arhitektura: dva stekovana GRU sloja + Dropout regularizacija + Dense izlazni sloj.
#       Ista arhitektura kao LSTM radi fer poredjenja.
#       Early Stopping na osnovu validation loss-a.
#
# Input:
#       - data/processed/X_train.pkl, X_val.pkl, X_test.pkl
#       - data/processed/y_train.pkl, y_val.pkl, y_test.pkl
#       - data/processed/dates_train.pkl, dates_val.pkl, dates_test.pkl
#       - data/processed/scaler.pkl
#
# Output:
#       - data/processed/gru_model.keras
#       - data/processed/gru_history.pkl
#       - data/processed/gru_predictions.pkl
#       - data/processed/gru_metrics.pkl

# %% biblioteke

import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %% ucitavanje podataka

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

X_train = load_pkl('data/processed/X_train.pkl')
X_val   = load_pkl('data/processed/X_val.pkl')
X_test  = load_pkl('data/processed/X_test.pkl')

y_train = load_pkl('data/processed/y_train.pkl')
y_val   = load_pkl('data/processed/y_val.pkl')
y_test  = load_pkl('data/processed/y_test.pkl')

dates_train = load_pkl('data/processed/dates_train.pkl')
dates_val   = load_pkl('data/processed/dates_val.pkl')
dates_test  = load_pkl('data/processed/dates_test.pkl')

scaler = load_pkl('data/processed/scaler.pkl')

print("Uspesno ucitani podaci:")
print(f"  X_train : {X_train.shape}  | y_train : {y_train.shape}")
print(f"  X_val   : {X_val.shape}   | y_val   : {y_val.shape}")
print(f"  X_test  : {X_test.shape}   | y_test  : {y_test.shape}")
print(f"  Oblik (Samples, Timesteps, Features): {X_train.shape}")

# %% definisanje parametara modela

GRU_UNITS_1    = 64
GRU_UNITS_2    = 32
DROPOUT_RATE   = 0.10
DENSE_UNITS    = 16
LEARNING_RATE  = 0.001
BATCH_SIZE     = 32
MAX_EPOCHS     = 150
PATIENCE       = 35
N_FEATURES     = X_train.shape[2]
N_TIMESTEPS    = X_train.shape[1]

print("\nParametri modela:")
print(f"  Timesteps : {N_TIMESTEPS}")
print(f"  Features  : {N_FEATURES}")
print(f"  GRU sloj 1: {GRU_UNITS_1} neurona")
print(f"  GRU sloj 2: {GRU_UNITS_2} neurona")
print(f"  Dropout   : {DROPOUT_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epoha : {MAX_EPOCHS}")
print(f"  Patience  : {PATIENCE}")

# %% izgradnja arhitekture modela

tf.random.set_seed(42)
np.random.seed(42)

model = Sequential([
    # Prvi GRU sloj - return_sequences=True jer sledi jos jedan GRU sloj
    GRU(GRU_UNITS_1,
        return_sequences=True,
        kernel_initializer='glorot_uniform',
        input_shape=(N_TIMESTEPS, N_FEATURES),
        name='gru_1'),
    Dropout(DROPOUT_RATE, name='dropout_1'),

    # Drugi GRU sloj - return_sequences=False jer sledi Dense sloj
    GRU(GRU_UNITS_2,
        return_sequences=False,
        kernel_initializer='glorot_uniform',
        name='gru_2'),
    Dropout(DROPOUT_RATE, name='dropout_2'),

    Dense(DENSE_UNITS, activation='relu', kernel_initializer='glorot_uniform', name='dense_1'),

    # Izlazni sloj
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

print("\nPokretanje treniranja GRU modela...")

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

plt.suptitle('GRU - Krive ucenja', fontsize=13)
plt.tight_layout()
plt.savefig('results/gru_learning_curve.png', dpi=150, bbox_inches='tight')
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
axes[0].plot(dates_val_dt, y_val_pred, color='red', linewidth=1.5, linestyle='--', label='GRU predikcije')
axes[0].fill_between(dates_val_dt, y_val_true, y_val_pred, alpha=0.15, color='red', label='Greška')
axes[0].set_title(f'Validacioni skup | RMSE={val_rmse:.4f} | MAE={val_mae:.4f} | MAPE={val_mape:.2f}%')
axes[0].set_ylabel('USD/galon')
axes[0].legend()

axes[1].plot(dates_test_dt, y_test_true, color='steelblue', linewidth=1.5, label='Stvarne vrednosti')
axes[1].plot(dates_test_dt, y_test_pred, color='red', linewidth=1.5, linestyle='--', label='GRU predikcije')
axes[1].fill_between(dates_test_dt, y_test_true, y_test_pred, alpha=0.15, color='red', label='Greška')
axes[1].set_title(f'Test skup | RMSE={test_rmse:.4f} | MAE={test_mae:.4f} | MAPE={test_mape:.2f}%')
axes[1].set_ylabel('USD/galon')
axes[1].set_xlabel('Datum')
axes[1].legend()

plt.suptitle('GRU - Predikcije vs Stvarne vrednosti', fontsize=13)
plt.tight_layout()
plt.savefig('results/gru_predikcije.png', dpi=150, bbox_inches='tight')
plt.show()

# %% residual plot

reziduali_test = y_test_true - y_test_pred

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

axes[0].plot(dates_test_dt, reziduali_test, color='darkorange', linewidth=0.8)
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0].set_title('Reziduali na test skupu (stvarna - predviđena vrednost)')
axes[0].set_ylabel('Greska (USD/galon)')

axes[1].hist(reziduali_test, bins=40, color='darkorange', alpha=0.7, edgecolor='white')
axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_title('Distribucija reziduala')
axes[1].set_xlabel('Greska (USD/galon)')
axes[1].set_ylabel('Frekvencija')

plt.tight_layout()
plt.savefig('results/gru_reziduali.png', dpi=150, bbox_inches='tight')
plt.show()

# %% cuvanje modela i rezultata

gru_metrics = {
    'model'     : 'GRU',
    'val_rmse'  : val_rmse,
    'val_mae'   : val_mae,
    'val_mape'  : val_mape,
    'test_rmse' : test_rmse,
    'test_mae'  : test_mae,
    'test_mape' : test_mape,
}

gru_predictions = {
    'dates_test'  : dates_test_dt,
    'y_true'      : y_test_true,
    'y_pred'      : y_test_pred,
    'dates_val'   : dates_val_dt,
    'val_true'    : y_val_true,
    'val_pred'    : y_val_pred,
}

model.save('data/processed/gru_model.keras')

with open('data/processed/gru_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

with open('data/processed/gru_predictions.pkl', 'wb') as f:
    pickle.dump(gru_predictions, f)

with open('data/processed/gru_metrics.pkl', 'wb') as f:
    pickle.dump(gru_metrics, f)

print("\nSacuvano:")
print("data/processed/gru_model.keras")
print("data/processed/gru_history.pkl")
print("data/processed/gru_predictions.pkl")
print("data/processed/gru_metrics.pkl")
print("GRU TRENIRANJE ZAVRSENO")
print(f"  Val  RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | MAPE: {val_mape:.2f}%")
print(f"  Test RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | MAPE: {test_mape:.2f}%")

# %% metrike
# GRU TRENIRANJE ZAVRSENO
#  Val  RMSE: 0.0883 | MAE: 0.0762 | MAPE: 3.07%
#  Test RMSE: 0.2171 | MAE: 0.1824 | MAPE: 8.04%
