# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:17:51 2026

@author: danij
"""

# %% dokumentacija
 
# Opis: Diebold-Mariano test za statisticko poredjenje tacnosti svih 6 modela.
#       Ucitavamo sacuvane predikcije, poravnavamo na zajednicki test period
#       i racunamo DM test za sve parove modela.
#
# Input:
#       - data/processed/arima_predictions.pkl
#       - data/processed/arimax_predictions.pkl
#       - data/processed/lstm_predictions.pkl
#       - data/processed/gru_predictions.pkl
#       - data/processed/lstm_mix_predictions.pkl
#       - data/processed/gru_mix_predictions.pkl
#
# Output:
#       - data/processed/dm_rezultati.csv
 
# %% biblioteke
 
import pandas as pd
import numpy as np
import os
import pickle
import warnings
from itertools import combinations
from scipy import stats
 
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
 
# %% pomocne funkcije
 
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
 
# Diebold-Mariano test
# H0: dva modela imaju jednaku tacnost predvidjanja (E[d_t] = 0)
# H1: modeli se statisticki znacajno razlikuju po tacnosti
# Gubitna funkcija: kvadratna greska
# Vraca: dm_stat, p_vrednost (dvostrana), zakljucak
 
def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1):
    e1 = (y_true - y_pred1) ** 2
    e2 = (y_true - y_pred2) ** 2
 
    d = e1 - e2
    n = len(d)
    d_mean = np.mean(d)
 
    # Newey-West korekcija varijanse
    gamma = np.zeros(h)
    for lag in range(h):
        if lag == 0:
            gamma[lag] = np.mean((d - d_mean) ** 2)
        else:
            gamma[lag] = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
 
    var_d = (gamma[0] + 2 * np.sum(gamma[1:])) / n
 
    if var_d <= 0:
        return np.nan, np.nan, "Nije moguce izracunati (var <= 0)"
 
    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * stats.t.sf(np.abs(dm_stat), df=n - 1)
 
    if p_value < 0.01:
        nivo = "*** (p < 0.01)"
    elif p_value < 0.05:
        nivo = "** (p < 0.05)"
    elif p_value < 0.10:
        nivo = "* (p < 0.10)"
    else:
        nivo = "nije znacajno"
 
    bolji = "Model 2 je bolji" if dm_stat > 0 else "Model 1 je bolji"
    verdict = f"{nivo} | {bolji}"
 
    return dm_stat, p_value, verdict

# Znacenje zvezdica (nivo statisticke znacajnosti):
# *** (p < 0.01) - 99% sigurnosti da se modeli stvarno razlikuju po tacnosti
# **  (p < 0.05) - 95% sigurnosti
# *   (p < 0.10) - 90% sigurnosti
# bez zvezdice  - razlika nije znacajna, mogla je nastati slucajno
 
# %% ucitavanje predikcija
 
print("Ucitavanje predikcija:")
 
arima_pred    = load_pkl('data/processed/arima_predictions.pkl')
arimax_pred   = load_pkl('data/processed/arimax_predictions.pkl')
lstm_pred     = load_pkl('data/processed/lstm_predictions.pkl')
gru_pred      = load_pkl('data/processed/gru_predictions.pkl')
lstm_mix_pred = load_pkl('data/processed/lstm_mix_predictions.pkl')
gru_mix_pred  = load_pkl('data/processed/gru_mix_predictions.pkl')
 
# ARIMA i ARIMAX: kljuc 'dates',  ostali: kljuc 'dates_test'
def standardizuj(pred_dict, dates_key='dates'):
    dates = pd.to_datetime(pred_dict[dates_key])
    return {
        'dates'  : dates,
        'y_true' : np.array(pred_dict['y_true']),
        'y_pred' : np.array(pred_dict['y_pred']),
    }
 
modeli_raw = {
    'ARIMA'      : standardizuj(arima_pred,    dates_key='dates'),
    'ARIMAX'     : standardizuj(arimax_pred,   dates_key='dates'),
    'LSTM'       : standardizuj(lstm_pred,     dates_key='dates_test'),
    'GRU'        : standardizuj(gru_pred,      dates_key='dates_test'),
    'LSTM_mixed' : standardizuj(lstm_mix_pred, dates_key='dates_test'),
    'GRU_mixed'  : standardizuj(gru_mix_pred,  dates_key='dates_test'),
}
 
for naziv, d in modeli_raw.items():
    print(f"  {naziv:<12}: {len(d['y_pred'])} uzoraka | "
          f"{d['dates'].min().date()} -> {d['dates'].max().date()}")
 
# %% poravnavanje na zajednicki test period
 
print("Poravnanje na zajednicki datumski opseg")
 
svi_datumi = [set(d['dates'].date) for d in modeli_raw.values()]
zajednicki_datumi = sorted(svi_datumi[0].intersection(*svi_datumi[1:]))
zajednicki_datumi = pd.to_datetime(zajednicki_datumi)
 
print(f"Zajednicki period: {zajednicki_datumi.min().date()} -> {zajednicki_datumi.max().date()}")
print(f"Broj zajednickih tacaka: {len(zajednicki_datumi)}")
 
modeli = {}
for naziv, d in modeli_raw.items():
    maska = d['dates'].isin(zajednicki_datumi)
    modeli[naziv] = {
        'dates'  : d['dates'][maska],
        'y_true' : d['y_true'][maska],
        'y_pred' : d['y_pred'][maska],
    }
    print(f"  {naziv:<12}: {maska.sum()} uzoraka zadrzano")
 
# zajednicki y_true (prosek radi konzistentnosti)
y_true_zajednicki = np.mean([d['y_true'] for d in modeli.values()], axis=0)
 
# %% Diebold-Mariano test
 
print("DIEBOLD-MARIANO TEST (svi parovi modela)")
print("H0: modeli imaju jednaku tacnost predvidjanja")
print("Zajednicki test period | Funkcija gubitka: kvadratna greska")
print()
print(f"{'Par modela':<30} {'DM stat':>10} {'p-vrednost':>12} {'Zakljucak'}")
 
nazivi_modela = list(modeli.keys())
dm_rezultati = []
 
for naziv1, naziv2 in combinations(nazivi_modela, 2):
    y_pred1 = modeli[naziv1]['y_pred']
    y_pred2 = modeli[naziv2]['y_pred']
 
    dm_stat, p_val, verdict = diebold_mariano_test(
        y_true_zajednicki, y_pred1, y_pred2, h=1
    )
 
    dm_rezultati.append({
        'Model 1'    : naziv1,
        'Model 2'    : naziv2,
        'DM stat'    : round(dm_stat, 4) if not np.isnan(dm_stat) else np.nan,
        'p-vrednost' : round(p_val, 4)   if not np.isnan(p_val)   else np.nan,
        'Zakljucak'  : verdict,
    })
 
    par = f"{naziv1} vs {naziv2}"
    if not np.isnan(dm_stat):
        print(f"  {par:<28} {dm_stat:>10.4f} {p_val:>12.4f}   {verdict}")
    else:
        print(f"  {par:<28}   {'N/A':>10}   {'N/A':>10}   {verdict}")
 
# %% cuvanje rezultata
 
df_dm = pd.DataFrame(dm_rezultati)
df_dm.to_csv('data/processed/dm_rezultati.csv', index=False)
print("\nSacuvano: data/processed/dm_rezultati.csv")

# %% rezultati DM testa
#
# Par modela                        DM stat   p-vrednost   Zakljucak
# ARIMA vs ARIMAX                    0.2442       0.8086   nije znacajno | Model 2 je bolji
# ARIMA vs LSTM                     -8.5359       0.0000   *** (p < 0.01) | Model 1 je bolji
# ARIMA vs GRU                      -7.4688       0.0000   *** (p < 0.01) | Model 1 je bolji
# ARIMA vs LSTM_mixed              -10.9741       0.0000   *** (p < 0.01) | Model 1 je bolji
# ARIMA vs GRU_mixed                -6.1486       0.0000   *** (p < 0.01) | Model 1 je bolji
# ARIMAX vs LSTM                    -8.6458       0.0000   *** (p < 0.01) | Model 1 je bolji
# ARIMAX vs GRU                     -7.4795       0.0000   *** (p < 0.01) | Model 1 je bolji
# ARIMAX vs LSTM_mixed             -10.9832       0.0000   *** (p < 0.01) | Model 1 je bolji
# ARIMAX vs GRU_mixed               -6.1948       0.0000   *** (p < 0.01) | Model 1 je bolji
# LSTM vs GRU                        0.6885       0.4958   nije znacajno | Model 2 je bolji
# LSTM vs LSTM_mixed                -8.1757       0.0000   *** (p < 0.01) | Model 1 je bolji
# LSTM vs GRU_mixed                 -4.1784       0.0002   *** (p < 0.01) | Model 1 je bolji
# GRU vs LSTM_mixed                 -8.1269       0.0000   *** (p < 0.01) | Model 1 je bolji
# GRU vs GRU_mixed                  -4.1362       0.0002   *** (p < 0.01) | Model 1 je bolji
# LSTM_mixed vs GRU_mixed            4.2594       0.0002   *** (p < 0.01) | Model 2 je bolji