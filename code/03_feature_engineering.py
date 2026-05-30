# -*- coding: utf-8 -*-
"""
Created on Sat May 30 10:42:48 2026

@author: danij
"""

# %% dokumentacija

# Opis: Feature Engineering i priprema za Supervised Learning
#       Kreiranje lagged features kao dodatnih prediktora za LSTM i GRU.
#       Izracunavanje rolling statistics za prozoer sirine 4 i 12 nedelja.
#       Sliding Window
#       Hronoloska podela skupa (70/15/15)
#
# Logika: ARIMA radi direktno na originalnoj seriji i sama hendluje lagove.
#         LSTM i GRU nemaju urodjen koncept tabele sa istorijom, pa im rucno 
#         saljemo prosle vrednosti nafte, dolara i benzina da bi predvideli buducnost.
#
# Output:
#       - data/processed/df_features.csv
#       - data/processed/df_features_clean.csv
#       - data/processed/X_lstm.pkl, y_lstm.pkl
#       - data/processed/X_train.pkl, X_val.pkl, X_test.pkl
#       - data/processed/y_train.pkl, y_val.pkl, y_test.pkl
#       - data/processed/dates_train.pkl, dates_val.pkl, dates_test.pkl
# %% biblioteke
 
import pandas as pd
import numpy as np
import os
import pickle
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %% ucitavanje podataka

# df_scaled jer su ove kolone direktan ulaz u neuronske mreze
df_scaled = pd.read_csv('../data/processed/df_scaled.csv', parse_dates=['date'])

print("Uspesno ucitan df_scaled:")
print(f"Dimenzije pre filtriranja: {df_scaled.shape}")

# %% definisanje parametara za feature engineering

LAG_WEEKS = [1, 4, 8]
ROLLING_WINDOWS = [4, 12]

FEATURE_COLS = ['regular_conv', 'crude_oil', 'usd_index']

# %% kreiranje lagged features

# Pravimo kopiju da bismo ocuvali df_scaled netaknutim
df_fe = df_scaled.copy()

print("\n--- Pokretanje generisanja lagovanih kolona ---")
for col in FEATURE_COLS:
    for lag in LAG_WEEKS:
        naziv_kolone = f'{col}_lag{lag}'
        
        # shift(lag) pomera podatke nadole za lag broj mesta
        # U redu za nedelju T, lag1 sadrzi podatak iz nedelje T-1
        df_fe[naziv_kolone] = df_fe[col].shift(lag)
        
        print(f"Kreirana kolona: {naziv_kolone}")
        
# %% izracunavanje rolling statistics

print("\nPokretanje proracuna rolling statistika")
for col in FEATURE_COLS:
    for window in ROLLING_WINDOWS:
        # 1. Pokretni prosek (Rolling Mean)
        mean_col_name = f'{col}_roll_mean_{window}'
        df_fe[mean_col_name] = df_fe[col].rolling(window=window).mean().shift(1)
        
        # 2. Pokretna standardna devijacija (Rolling Std)
        std_col_name = f'{col}_roll_std_{window}'
        df_fe[std_col_name] = df_fe[col].rolling(window=window).std().shift(1)
        
        print(f"Kreirane rolling kolone (mean & std) za: {col} (prozor: {window} nedelja)")
        
# %% pregled svih generisanih podataka

print("\nSve kolone u novom DataFrame-u:")
print(df_fe.columns.tolist())

preview_cols = ['date', 'regular_conv', 'regular_conv_lag1', 'regular_conv_roll_mean_4', 'regular_conv_roll_std_4',
                'crude_oil', 'crude_oil_lag1', 'crude_oil_roll_mean_12', 'usd_index', 'usd_index_lag1']

print("\nPregled prvih 15 redova:")
print(df_fe[preview_cols].head(15).to_string())

# Provera NaN vrednosti za sve novokreirane kolone
nove_kolone = [c for c in df_fe.columns if '_lag' in c or '_roll_' in c]
print("\nBroj NaN vrednosti po novim kolonama:")
print(df_fe[nove_kolone].isnull().sum())

# %% cuvanje podataka

df_fe.to_csv('../data/processed/df_features.csv', index=False)

print("Podaci sacuvani na putanji: data/processed/df_features.csv")
print(f"Konacne dimenzije tabele: {df_fe.shape}")

# %% ciscenje NaN vrednosti
 
# dropna() uklanja sve redove koji imaju barem jedan NaN.
df_clean = df_fe.dropna().reset_index(drop=True)
print(f"Shape nakon dropna(): {df_clean.shape}")
print(f"Uklonjeno redova: {len(df_fe) - len(df_clean)}")
 
df_clean.to_csv('../data/processed/df_features_clean.csv', index=False)
print("Sacuvano: df_features_clean.csv")

# %% sliding window

print("\nPokretanje transformacije metodom klizeceg prozora")

# odvajanje kolona koje idu u matricu prediktora (X)
# izbacujemo datum i ostale kategorije goriva, za regular_conv imamo prethodne vrednosti,
# ako se ostavi ta kolona mreza ce vec znati unapred, zato nam sluze features koje smo racunali
izbaci_kolone = ['date', 'diesel', 'midgrade_ref', 'premium_conv', 'premium_ref', 'midgrade_conv', 'regular_ref', 'regular_conv']
X_cols = [col for col in df_clean.columns if col not in izbaci_kolone]

print(f"Broj features za LSTM/GRU: {len(X_cols)}")

# ciljna varijabla je cena regularnog benzina (nju predvidjamo)
target_col = 'regular_conv'

# pretvaramo podatke u NumPy nizove jer Keras/TensorFlow ne primaju direktno DataFrame
X_raw = df_clean[X_cols].values
y_raw = df_clean[target_col].values

# cuvamo i kolonu datuma za kasnije - koristicemo je za vizualizaciju predikcija
# da bismo znali koji datum odgovara kom sample-u
dates_raw = df_clean['date'].values

# funkcija za pakovanje podataka u 3D oblik (Samples, Timesteps, Features)
def create_sliding_window(X_data, y_data, dates_data, look_back=1):
    X_3D, y_3D, dates_3D  = [], [], []
    for i in range(len(X_data) - look_back):
        X_3D.append(X_data[i:(i + look_back), :])
        y_3D.append(y_data[i + look_back])
        # datum koji odgovara ovom sample-u je datum nedelje T+1 (ono sto predvidjamo)
        dates_3D.append(dates_data[i + look_back])
    return np.array(X_3D), np.array(y_3D), np.array(dates_3D)

# definisanje sirine look-back prozora (posmatramo prethodne 4 nedelje kao jedan korak)
LOOK_BACK = 4
X_lstm, y_lstm, dates_lstm = create_sliding_window(X_raw, y_raw, dates_raw, look_back=LOOK_BACK)

# Provera dimenzija krajnjih matrica za LSTM i GRU
print("\nFINALNE DIMENZIJE MATRICA ZA NEURONSKE MREZE")
print(f"X_lstm oblik (Samples, Timesteps, Features): {X_lstm.shape}")
print(f"y_lstm oblik (Samples, ): {y_lstm.shape}")

# %% hronoloska podela skupa (70 / 15 / 15)
 
# VAZNO: podela mora biti hronoloska (bez shuffle-a)!
 
n = len(X_lstm)
 
train_end = int(n * 0.70)
val_end   = int(n * 0.85)
 
X_train = X_lstm[:train_end]
X_val   = X_lstm[train_end:val_end]
X_test  = X_lstm[val_end:]
 
y_train = y_lstm[:train_end]
y_val   = y_lstm[train_end:val_end]
y_test  = y_lstm[val_end:]
 
dates_train = dates_lstm[:train_end]
dates_val   = dates_lstm[train_end:val_end]
dates_test  = dates_lstm[val_end:]
 
print("\nHronoloska podela skupa")
print(f"Ukupno samples: {n}")
print(f"Train : {len(X_train)} samples | period: {str(dates_train[0])[:10]} -> {str(dates_train[-1])[:10]}")
print(f"Val   : {len(X_val)}  samples | period: {str(dates_val[0])[:10]}   -> {str(dates_val[-1])[:10]}")
print(f"Test  : {len(X_test)}  samples | period: {str(dates_test[0])[:10]}  -> {str(dates_test[-1])[:10]}")

# %% cuvanje svih skupova u pickle fajlove
 
with open('../data/processed/X_lstm.pkl', 'wb') as f:
    pickle.dump(X_lstm, f)
with open('../data/processed/y_lstm.pkl', 'wb') as f:
    pickle.dump(y_lstm, f)
 
with open('../data/processed/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('../data/processed/X_val.pkl', 'wb') as f:
    pickle.dump(X_val, f)
with open('../data/processed/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
 
with open('../data/processed/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('../data/processed/y_val.pkl', 'wb') as f:
    pickle.dump(y_val, f)
with open('../data/processed/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
 
with open('../data/processed/dates_train.pkl', 'wb') as f:
    pickle.dump(dates_train, f)
with open('../data/processed/dates_val.pkl', 'wb') as f:
    pickle.dump(dates_val, f)
with open('../data/processed/dates_test.pkl', 'wb') as f:
    pickle.dump(dates_test, f)
 
print("\nSvi skupovi sacuvani u data/processed/")