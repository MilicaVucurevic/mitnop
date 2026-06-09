# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:16:40 2026

@author: danij
"""

# %% dokumentacija
 
# Opis: Feature Engineering i priprema pomesanog skupa za LSTM.
#       Umesto da uzmemo ceo skup hronoloski, uzimamo 70% redova iz svakog
#       posebnog perioda (crisis_flag, covid_flag, normalni period) kako bismo
#       obezbedili ravnomerno zastupljenost sva tri tipa trzisnih uslova u skupu.
#       Spojeni skup se sortira hronoloski i na njemu se rade sliding window
#       i hronoloska podela 70/15/15.
#
# Logika:
#       - crisis period  : redovi gde je crisis_flag == 1, uzimamo 70%
#       - covid period   : redovi gde je covid_flag == 1, uzimamo 70%
#       - normalni period: redovi gde su oba flaga == 0, uzimamo 70%
#       - spojeni skup se sortira hronoloski
#       - na spojenom skupu radimo sliding window pa 70/15/15 podelu
#
# Output:
#       - data/processed/df_features_mixed.csv
#       - data/processed/X_train_mix.pkl, X_val_mix.pkl, X_test_mix.pkl
#       - data/processed/y_train_mix.pkl, y_val_mix.pkl, y_test_mix.pkl
#       - data/processed/dates_train_mix.pkl, dates_val_mix.pkl, dates_test_mix.pkl
 
# %% biblioteke
 
import pandas as pd
import numpy as np
import os
import pickle
 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
 
# %% ucitavanje podataka
 
df_scaled = pd.read_csv('data/processed/df_scaled.csv', parse_dates=['date'])
 
print("Uspesno ucitan df_scaled:")
print(f"Dimenzije: {df_scaled.shape}")
 
# %% definisanje parametara
 
LAG_WEEKS      = [1, 4, 8]
ROLLING_WINDOWS = [4, 12]
FEATURE_COLS   = ['regular_conv', 'crude_oil', 'usd_index']
LOOK_BACK      = 4
 
# %% kreiranje lagged features i rolling statistika
 
df_fe = df_scaled.copy()
 
print("\nKreiranje lagovanih kolona")
for col in FEATURE_COLS:
    for lag in LAG_WEEKS:
        df_fe[f'{col}_lag{lag}'] = df_fe[col].shift(lag)
        print(f"Kreirana kolona: {col}_lag{lag}")
 
print("\nKreiranje rolling statistika")
for col in FEATURE_COLS:
    for window in ROLLING_WINDOWS:
        df_fe[f'{col}_roll_mean_{window}'] = df_fe[col].rolling(window=window).mean().shift(1)
        df_fe[f'{col}_roll_std_{window}']  = df_fe[col].rolling(window=window).std().shift(1)
        print(f"Kreirane rolling kolone za: {col} (prozor: {window} nedelja)")
 
# ciscenje NaN vrednosti (nastaju zbog shift i rolling operacija)
df_clean = df_fe.dropna().reset_index(drop=True)
print(f"\nShape nakon dropna(): {df_clean.shape}")
 
# %% mesanje - uzimamo 70% iz svakog perioda posebno
 
print("\nMesanje po periodima")
 
# razdvajamo tri perioda
df_crisis  = df_clean[df_clean['crisis_flag'] == 1].copy()
df_covid   = df_clean[df_clean['covid_flag']  == 1].copy()
df_normalni = df_clean[(df_clean['crisis_flag'] == 0) & 
                        (df_clean['covid_flag']  == 0)].copy()
 
print(f"Crisis period  : {len(df_crisis)} redova")
print(f"COVID period   : {len(df_covid)} redova")
print(f"Normalni period: {len(df_normalni)} redova")
 
# uzimamo 70% iz svakog perioda hronoloski
n_crisis   = int(len(df_crisis)   * 0.70)
n_covid    = int(len(df_covid)    * 0.70)
n_normalni = int(len(df_normalni) * 0.70)
 
df_crisis_70   = df_crisis.iloc[:n_crisis]
df_covid_70    = df_covid.iloc[:n_covid]
df_normalni_70 = df_normalni.iloc[:n_normalni]
 
print("\nUzeto 70% iz svakog perioda:")
print(f"  Finansijska kriza  : {len(df_crisis_70)} redova | {df_crisis_70['date'].iloc[0].date()} -> {df_crisis_70['date'].iloc[-1].date()}")
print(f"  COVID   : {len(df_covid_70)} redova  | {df_covid_70['date'].iloc[0].date()} -> {df_covid_70['date'].iloc[-1].date()}")
print(f"  Normalni: {len(df_normalni_70)} redova | {df_normalni_70['date'].iloc[0].date()} -> {df_normalni_70['date'].iloc[-1].date()}")
 
# spajamo i sortiramo hronoloski
df_mix = pd.concat([df_crisis_70, df_covid_70, df_normalni_70])
df_mix = df_mix.sort_values('date').reset_index(drop=True)
 
print(f"\nSpojeni skup: {len(df_mix)} redova")
print(f"Period: {df_mix['date'].iloc[0].date()} -> {df_mix['date'].iloc[-1].date()}")
 
# cuvamo za pregled
df_mix.to_csv('data/processed/df_features_mixed.csv', index=False)
print("Sacuvano: df_features_mixed.csv")
 
# %% sliding window na spojenom skupu
 
print("\nSliding window")
 
izbaci_kolone = ['date', 'diesel', 'midgrade_ref', 'premium_conv', 'premium_ref',
                 'midgrade_conv', 'regular_ref', 'regular_conv']
X_cols = [col for col in df_mix.columns if col not in izbaci_kolone]
 
print(f"Broj features: {len(X_cols)}")
 
X_raw    = df_mix[X_cols].values
y_raw    = df_mix['regular_conv'].values
dates_raw = df_mix['date'].values
 
def create_sliding_window(X_data, y_data, dates_data, look_back=1):
    X_3D, y_3D, dates_3D = [], [], []
    for i in range(len(X_data) - look_back):
        X_3D.append(X_data[i:(i + look_back), :])
        y_3D.append(y_data[i + look_back])
        dates_3D.append(dates_data[i + look_back])
    return np.array(X_3D), np.array(y_3D), np.array(dates_3D)
 
X_lstm, y_lstm, dates_lstm = create_sliding_window(X_raw, y_raw, dates_raw, look_back=LOOK_BACK)
 
print("\nFinalne dimenzije:")
print(f"X_lstm oblik (Samples, Timesteps, Features): {X_lstm.shape}")
print(f"y_lstm oblik (Samples, ): {y_lstm.shape}")
 
# %% hronoloska podela spojenog skupa (70 / 15 / 15)
 
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
 
print("\nHronoloska podela spojenog skupa (70/15/15):")
print(f"Ukupno uzoraka: {n}")
print(f"Train : {len(X_train)} uzoraka | period: {str(dates_train[0])[:10]} -> {str(dates_train[-1])[:10]}")
print(f"Val   : {len(X_val)}  uzoraka | period: {str(dates_val[0])[:10]}   -> {str(dates_val[-1])[:10]}")
print(f"Test  : {len(X_test)}  uzoraka | period: {str(dates_test[0])[:10]}  -> {str(dates_test[-1])[:10]}")
 
# %% cuvanje skupova
 
with open('data/processed/X_train_mix.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('data/processed/X_val_mix.pkl', 'wb') as f:
    pickle.dump(X_val, f)
with open('data/processed/X_test_mix.pkl', 'wb') as f:
    pickle.dump(X_test, f)
 
with open('data/processed/y_train_mix.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('data/processed/y_val_mix.pkl', 'wb') as f:
    pickle.dump(y_val, f)
with open('data/processed/y_test_mix.pkl', 'wb') as f:
    pickle.dump(y_test, f)
 
with open('data/processed/dates_train_mix.pkl', 'wb') as f:
    pickle.dump(dates_train, f)
with open('data/processed/dates_val_mix.pkl', 'wb') as f:
    pickle.dump(dates_val, f)
with open('data/processed/dates_test_mix.pkl', 'wb') as f:
    pickle.dump(dates_test, f)
 
print("\nSvi skupovi sacuvani u data/processed/")