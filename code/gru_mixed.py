# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:35:10 2026

@author: danij
"""

# %% dokumentacija
 
# Opis: Razvoj i treniranje GRU modela na izmesanom skupu podataka.
#       70% iz perioda finansijske krize + 70% iz covid perioda +
#       70% iz normalnog perioda, spojeno hronoloski.
#       Arhitektura je identicna originalnom GRU modelu:
#       dva GRU sloja + Dropout + Dense izlazni sloj.
#       Cilj: ispitati da li ravnomerna zastupljenost sva tri tipa trzisnih
#       uslova u trening skupu poboljsava generalizaciju GRU modela.
#
# Input:
#       - data/processed/X_train_mix.pkl, X_val_mix.pkl, X_test_mix.pkl
#       - data/processed/y_train_mix.pkl, y_val_mix.pkl, y_test_mix.pkl
#       - data/processed/dates_train_mix.pkl, dates_val_mix.pkl, dates_test_mix.pkl
#       - data/processed/scaler.pkl
#
# Output:
#       - data/processed/gru_mix_model.keras
#       - data/processed/gru_mix_history.pkl
#       - data/processed/gru_mix_predictions.pkl
#       - data/processed/gru_mix_metrics.pkl
 
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
 
X_train = load_pkl('../data/processed/X_train_mix.pkl')
X_val   = load_pkl('../data/processed/X_val_mix.pkl')
X_test  = load_pkl('../data/processed/X_test_mix.pkl')
 
y_train = load_pkl('../data/processed/y_train_mix.pkl')
y_val   = load_pkl('../data/processed/y_val_mix.pkl')
y_test  = load_pkl('../data/processed/y_test_mix.pkl')
 
dates_train = load_pkl('../data/processed/dates_train_mix.pkl')
dates_val   = load_pkl('../data/processed/dates_val_mix.pkl')
dates_test  = load_pkl('../data/processed/dates_test_mix.pkl')
 
scaler = load_pkl('../data/processed/scaler.pkl')
 
print("Uspesno ucitani podaci (pomesani skup):")
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
    GRU(GRU_UNITS_1,
        return_sequences=True,
        kernel_initializer='glorot_uniform',
        input_shape=(N_TIMESTEPS, N_FEATURES),
        name='gru_1'),
    Dropout(DROPOUT_RATE, name='dropout_1'),
 
    GRU(GRU_UNITS_2,
        return_sequences=False,
        kernel_initializer='glorot_uniform',
        name='gru_2'),
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
 
print("\nPokretanje treniranja GRU modela (izmesani skup)...")
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
 
plt.suptitle('GRU (izmesani skup) - Krive ucenja', fontsize=13)
plt.tight_layout()
#plt.savefig('../../data/processed/gru_mix_learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# Loss (MSE) — levi grafik:
# I train i val loss naglo padaju u prvim epohama, model brzo uci osnovne obrasce.
# Nakon toga oba se stabilizuju i ostaju blizu jedan drugom kroz ceo trening.
# Nema velikog raskoraka izmedju train i val loss, nema znakova overfittinga.

# MAE — desni grafik:
# Isti obrazac, nagli pad pa stabilizacija.
# Val MAE (crvena) osciluje oko train MAE (plava), oscilacije su prisutne ali nisu dramaticne.

# Zakljucak:
# Model konvergira stabilno, bez znakova overfittinga.
# Trening se zaustavio oko epohe 90, sto znaci da je Early Stopping
# pronasao dobru tacku zaustavljanja.
# Ponasanje kriva je konzistentno sa ocekivanjima za GRU na malom, izmesanom skupu.
 
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
 
plt.suptitle('GRU (izmesani skup) - Predikcije vs Stvarne vrednosti', fontsize=13)
plt.tight_layout()
#plt.savefig('../../data/processed/gru_mix_predikcije.png', dpi=150, bbox_inches='tight')
plt.show()

# Vizualizacija predikcija - gornji grafik (Validacioni skup):
# Model solidno prati opsti trend kretanja cena.

# Vizualizacija predikcija — donji grafik (Test skup):
# Model dobro prati trend u stabilnim periodima (2016-2019).
# Vidljiv problem u periodu naglog rasta cena (2017-2018), model potcenjuje vrednosti,
# stvarne cene rastu brze nego sto model predvidja.
# Najveca greska je u 2020. godini, COVID pad je iznenadio model,
# sto je ocekivano jer su ovakvi ekstremni dogadjaji tesko predvidivi.
# model se brzo vraca na pravi trend nakon pada
 
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
#plt.savefig('../../data/processed/gru_mix_reziduali.png', dpi=150, bbox_inches='tight')
plt.show()

# Gornji grafik - Reziduali kroz vreme:
# 2015-2016: reziduali negativni i osciluju, model precenjuje cene na pocetku test perioda.
# 2016-2018: reziduali blizu nule sa manjim oscilacijama, model se dobro snalazi
#            u stabilnom periodu.
# 2018-2020: reziduali postaju konzistentno pozitivni i blago rastu,
#            model potcenjuje cene tokom perioda rasta, stvarne cene rastu
#            brze nego sto model predvidja.
# 2020: COVID pad iznenadio model
# Postoji blagi trend rasta reziduala kroz vreme

# Donji grafik — Distribucija reziduala:
# Distribucija je asimetricna — reziduali su vise koncentrisani na pozitivnoj strani,
# sto potvrdjuje da model potcenjuje cene, posebno u periodu rasta.
# Raspon gresaka je od oko -0.30 do +0.25 USD/galonu.
# Distribucija nije centrirana oko nule, sto znaci da postoji blaga u predikcijama.
# Slicno sa rezidualima izmesanog LSTM-a, oba modela pokazuju sličan
# obrazac potcenjivanja u periodima rasta cena.
 
# %% cuvanje modela i rezultata
 
gru_mix_metrics = {
    'model'     : 'GRU_mixed',
    'val_rmse'  : val_rmse,
    'val_mae'   : val_mae,
    'val_mape'  : val_mape,
    'test_rmse' : test_rmse,
    'test_mae'  : test_mae,
    'test_mape' : test_mape,
}
 
gru_mix_predictions = {
    'dates_test'  : dates_test_dt,
    'y_true'      : y_test_true,
    'y_pred'      : y_test_pred,
    'dates_val'   : dates_val_dt,
    'val_true'    : y_val_true,
    'val_pred'    : y_val_pred,
}
 
os.makedirs('../../data/processed', exist_ok=True)
model.save('../../data/processed/gru_mix_model.keras')
 
with open('../../data/processed/gru_mix_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
 
with open('../../data/processed/gru_mix_predictions.pkl', 'wb') as f:
    pickle.dump(gru_mix_predictions, f)
 
with open('../../data/processed/gru_mix_metrics.pkl', 'wb') as f:
    pickle.dump(gru_mix_metrics, f)
 
print("\nSacuvano:")
print("  data/processed/gru_mix_model.keras")
print("  data/processed/gru_mix_history.pkl")
print("  data/processed/gru_mix_predictions.pkl")
print("  data/processed/gru_mix_metrics.pkl")
print("GRU (IZMESANI SKUP) TRENIRANJE ZAVRSENO")
print(f"  Val  RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | MAPE: {val_mape:.2f}%")
print(f"  Test RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | MAPE: {test_mape:.2f}%")

# %% metrike

# Metrike na VALIDACIONOM skupu:
#  RMSE : 0.1766 USD/galon
#  MAE  : 0.1492 USD/galon
#  MAPE : 5.58%

# Metrike na TEST skupu:
#  RMSE : 0.1246 USD/galon
#  MAE  : 0.1044 USD/galon
#  MAPE : 5.37%

# Na validacionom skupu model gresi prosecno oko 17-18 centi po galonu (MAPE 5.58%),
# sto je slabije od originalnog GRU-a (MAPE 3.07%) — ocekivano jer je validacioni
# skup drugacije sastavljen.
# Na test skupu model gresi prosecno oko 12-13 centi po galonu (MAPE 5.37%),
# sto je znacajno bolje od originalnog GRU-a (MAPE 8.04%).
# Obrazac je isti kao kod izmesanog LSTM-a — losije na validaciji, bolje na testu.
# Ravnomerna zastupljenost sva tri trzisna perioda u trening skupu pomogla je
# modelu da se bolje generalizuje na nevidjenim podacima.
# GRU izmesani vs LSTM izmesani: metrike su prakticno identicne
# (GRU test RMSE 0.1246 vs LSTM 0.1260), sto potvrdjuje da arhitektura
# nije kljucni faktor, vaznije je kako je skup sastavljen.