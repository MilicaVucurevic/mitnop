# -*- coding: utf-8 -*-
"""
Created on Sun May 31 14:47:26 2026

@author: Milica
"""

# %% dokumentacija
 
# Opis: Razvoj i treniranje ARIMA modela za predikciju cena regular_conv benzina.
#       ARIMA je univarijantna metoda - koristi samo istorijske cene goriva (1995-2021).
#       Parametar d=1 utvrdjen ADF testom u 02_eda.py.
#       Parametri p i q odredjuju se na osnovu ACF/PACF grafika.
#       Walk-Forward validacija za realnu procenu generalizacije.
#
# Input:
#       - data/processed/df_original.csv
#
# Output:
#       - data/processed/arima_predictions.pkl
#       - data/processed/arima_model.pkl
#       - data/processed/arima_metrics.pkl

# %% biblioteke
 
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
 
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# %% ucitavanje podataka
 
# ARIMA koristi originalne (neskalirane) vrednosti na punom periodu 1995-2021
df = pd.read_csv('data/processed/df_original.csv', parse_dates=['date'])
df = df.set_index('date')
 
serija = df['regular_conv'].dropna()
 
print(f"Ukupno redova: {len(serija)}")
print(f"Period: {serija.index.min().date()} -> {serija.index.max().date()}")
print(f"Min: {serija.min():.3f} | Max: {serija.max():.3f} | Mean: {serija.mean():.3f}")

# %% ACF i PACF grafici za odredjivanje p i q
 
# Nakon diferenciranja (d=1), gledamo:
#   PACF koji odredjuje p (AR deo)
#   ACF koji odredjuje q (MA deo)
 
serija_diff = serija.diff().dropna()
 
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
 
plot_acf(serija_diff, lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title('ACF - regular_conv (nakon 1. diferenciranja)')
axes[0].set_xlabel('Lag (nedelje)')
 
plot_pacf(serija_diff, lags=40, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('PACF - regular_conv (nakon 1. diferenciranja)')
axes[1].set_xlabel('Lag (nedelje)')
 
plt.tight_layout()
plt.savefig('results/acf_pacf.png', dpi=150, bbox_inches='tight')
plt.show()

# %% definisanje parametara modela
 
# Na osnovu ACF/PACF grafika biramo p i q.
# d=1 je vec utvrdjeno ADF testom.
 
p_vrednosti = [1, 2, 3]
q_vrednosti = [1, 2, 3, 4, 5]
d = 1
 
print("\nPretrazivanje optimalnih parametara (p, d, q) na osnovu AIC...")
print(f"{'p':>4} {'d':>4} {'q':>4} {'AIC':>12} {'BIC':>12}")
 

rezultati_grid = []
 
for p in p_vrednosti:
    for q in q_vrednosti:
        try:
            model_temp = ARIMA(serija, order=(p, d, q))
            fit_temp = model_temp.fit()
            rezultati_grid.append({
                'p': p, 'd': d, 'q': q,
                'aic': fit_temp.aic,
                'bic': fit_temp.bic
            })
            print(f"{p:>4} {d:>4} {q:>4} {fit_temp.aic:>12.2f} {fit_temp.bic:>12.2f}")
        except Exception as e:
            print(f"{p:>4} {d:>4} {q:>4}   GRESKA: {e}")
 
# sortiramo po AIC-u i uzimamo najbolji
df_grid = pd.DataFrame(rezultati_grid).sort_values('aic')
print("\nNajboljih 5 kombinacija (sortirano po AIC):")
print(df_grid.head())
 
best = df_grid.iloc[0]
BEST_P, BEST_D, BEST_Q = int(best['p']), int(best['d']), int(best['q'])
print(f"\nOdabrani parametri: ARIMA({BEST_P}, {BEST_D}, {BEST_Q})")

# %% treniranje finalnog ARIMA modela na trening skupu
 
# hronoloska podela 70/15/15 na originalnoj seriji (isti odnos kao za LSTM/GRU)
n = len(serija)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)
 
train_serija = serija.iloc[:train_end]
val_serija   = serija.iloc[train_end:val_end]
test_serija  = serija.iloc[val_end:]
 
print("\nPodela serije:")
print(f"  Train : {len(train_serija)} uzoraka | {train_serija.index[0].date()} -> {train_serija.index[-1].date()}")
print(f"  Val   : {len(val_serija)}  uzoraka | {val_serija.index[0].date()} -> {val_serija.index[-1].date()}")
print(f"  Test  : {len(test_serija)}  uzoraka | {test_serija.index[0].date()} -> {test_serija.index[-1].date()}")
 
# treniramo na trening skupu
print(f"\nTreniranje ARIMA({BEST_P}, {BEST_D}, {BEST_Q}) na trening skupu...")
arima_model = ARIMA(train_serija, order=(BEST_P, BEST_D, BEST_Q))
arima_fit = arima_model.fit()

val_pred_direct = arima_fit.forecast(steps=len(val_serija))
test_pred_direct_all = arima_fit.forecast(steps=len(val_serija) + len(test_serija))
test_pred_direct = test_pred_direct_all[len(val_serija):]

# metrike
val_rmse_d  = np.sqrt(mean_squared_error(val_serija.values, val_pred_direct))
val_mae_d   = mean_absolute_error(val_serija.values, val_pred_direct)
val_mape_d  = np.mean(np.abs((val_serija.values - val_pred_direct) / val_serija.values)) * 100

test_rmse_d  = np.sqrt(mean_squared_error(test_serija.values, test_pred_direct))
test_mae_d   = mean_absolute_error(test_serija.values, test_pred_direct)
test_mape_d  = np.mean(np.abs((test_serija.values - test_pred_direct) / test_serija.values)) * 100

print(f"Val  RMSE: {val_rmse_d:.4f} | MAE: {val_mae_d:.4f} | MAPE: {val_mape_d:.2f}%")
print(f"Test RMSE: {test_rmse_d:.4f} | MAE: {test_mae_d:.4f} | MAPE: {test_mape_d:.2f}%")
 

# %% dijagnostika reziduala
 
fig = arima_fit.plot_diagnostics(figsize=(14, 10))
fig.suptitle(f'Dijagnostika reziduala - ARIMA({BEST_P},{BEST_D},{BEST_Q})', fontsize=13)
plt.tight_layout()
plt.savefig('results/arima_dijagnostika.png', dpi=150, bbox_inches='tight')
plt.show()
 
# Ljung-Box test - da li su reziduali beli sum (bez autokorelacije)
lb_test = acorr_ljungbox(arima_fit.resid, lags=[10, 20], return_df=True)
print("\nLjung-Box test (p > 0.05 = reziduali su beli sum, model je dobar):")
print(lb_test)

# 1. grafik reziduala
#   - nema sablona, reziduali se krecu oko 0, ali imaju 2 velika skoka:
#    1. uragan Katrina (2005-2006)
#    2. finansijska kriza (2008-2009)

# 2. Histogram + KDE
# Plavi stubici (Hist) - stvarna distribucija reziduala
# Narandzasta kriva (KDE) - procenjena gustina reziduala
# Zelena kriva N(0,1) - idealna normalna raspodela
# narandzasta kriva je uza i visa od zelene, znaci da većina 
# reziduala je vrlo blizu nule (model cesto gresi malo), 
# ali povremeno pravi velike greske (oni skokovi iz prethodnog 
# grafika - Katrina, finansijska kriza).

# 3. Normal Q-Q plot
# srednji deo: tacke lepo prate crvenu liniju,
#              u normalnim trzisnim uslovima reziduali su normalno raspodeljeni
# levi rep: tacke skacu daleko ispod linije, posebno one tri 
#           izolovane tackice dole levo  - to su veliki padovi cena (Katrina 2005, finansijska kriza 2008) 
# desni rep: tacke idu iznad linije, ona jedna tacka gore desno - nagli skok cena

# 4. Correlogram
# Lag 0 je uvek 1.0, to je korelacija serije sa samom sobom
# Lagovi 1-10 - sve tacke su unutar plave zone
# Reziduali nemaju nikakvu preostalu autokorelaciju, svaka nedelja je
# nezavisna od prethodne. Model je iscrpeo svu informaciju iz serije,
# nije ostalo nista sto bi mogao da nauci.

# zakljucak: Sva cetiri grafika su konzistentna, ARIMA(3,1,2) je 
# solidan model. Jedini problem su ekstremni dogadjaji koji su ocekivani za ARIMA-u

# %% walk-forward validacija
 
# Walk-forward: model se postepeno prosiruje novim merenjima
# Simulira realno koriscenje - svake nedelje dobijamo novu stvarnu vrednost
# i azuriramo model pre sledece predikcije.
# Validacioni skup koristimo za procenu generalizacije pre test skupa.
 
print("\nWalk-Forward validacija na validacionom skupu...")
 
istorija = list(train_serija)
val_predictions = []
 
for t in range(len(val_serija)):
    model_wf = ARIMA(istorija, order=(BEST_P, BEST_D, BEST_Q))
    fit_wf = model_wf.fit()
    predikcija = fit_wf.forecast(steps=1)[0]
    val_predictions.append(predikcija)
    
    # dodajemo stvarnu vrednost u istoriju
    istorija.append(val_serija.iloc[t])
    
    if t % 50 == 0:
        print(f"  Korak {t+1}/{len(val_serija)}")
 
val_predictions = np.array(val_predictions)
 
val_rmse = np.sqrt(mean_squared_error(val_serija.values, val_predictions))
val_mae  = mean_absolute_error(val_serija.values, val_predictions)
val_mape = np.mean(np.abs((val_serija.values - val_predictions) / val_serija.values)) * 100
 
print("\nMetrike na VALIDACIONOM skupu:")
print(f"  RMSE : {val_rmse:.4f} USD/galon")
print(f"  MAE  : {val_mae:.4f} USD/galon")
print(f"  MAPE : {val_mape:.2f}%")

# RMSE 0.0432 — model prosecno gresi 4.3 centa po galonu
# MAE 0.0313 — tipicna greska je 3.1 cent po galonu
# MAPE 1.29% — greska je samo 1.29% od stvarne cene

# %% walk-forward predikcija na test skupu
 
print("\nWalk-Forward predikcija na test skupu...")
 
# prosirujemo istoriju i sa validacionim skupom pre testiranja
istorija_test = list(train_serija) + list(val_serija)
test_predictions = []
 
for t in range(len(test_serija)):
    model_wf = ARIMA(istorija_test, order=(BEST_P, BEST_D, BEST_Q))
    fit_wf = model_wf.fit()
    predikcija = fit_wf.forecast(steps=1)[0]
    test_predictions.append(predikcija)
    
    istorija_test.append(test_serija.iloc[t])
    
    if t % 50 == 0:
        print(f"  Korak {t+1}/{len(test_serija)}")
 
test_predictions = np.array(test_predictions)
 
test_rmse = np.sqrt(mean_squared_error(test_serija.values, test_predictions))
test_mae  = mean_absolute_error(test_serija.values, test_predictions)
test_mape = np.mean(np.abs((test_serija.values - test_predictions) / test_serija.values)) * 100
 
print("\nMetrike na TEST skupu:")
print(f"  RMSE : {test_rmse:.4f} USD/galon")
print(f"  MAE  : {test_mae:.4f} USD/galon")
print(f"  MAPE : {test_mape:.2f}%")

# test metrike su bolje od validacionih. To se desava jer test period
# (2019-2021) ukljucuje COVID, a walk-forward validacija postepeno 
# uci iz novih podataka, kada model vidi prve COVID vrednosti, 
# prilagodjava se i postaje bolji za sledece predikcije. 
# Validacioni skup (2016-2019) je bio stabilan period bez velikih sokova,
# ali i bez te adaptacije.

# %% vizualizacija predikcija na test skupu
 
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
 
# gornji grafik - ceo period sa istaknutim test skupom
axes[0].plot(serija.index, serija.values, color='steelblue', linewidth=0.8, label='Stvarne vrednosti')
axes[0].plot(test_serija.index, test_predictions, color='red', linewidth=1.2, 
             linestyle='--', label='ARIMA predikcije')
axes[0].axvline(x=test_serija.index[0], color='gray', linestyle=':', linewidth=1.5, label='Pocetak test skupa')
axes[0].set_title(f'ARIMA({BEST_P},{BEST_D},{BEST_Q}) - Predikcije vs Stvarne vrednosti (ceo period)')
axes[0].set_ylabel('USD/galon')
axes[0].legend()
 
# donji grafik - samo test period (uvecano)
axes[1].plot(test_serija.index, test_serija.values, color='steelblue', linewidth=1.5, label='Stvarne vrednosti')
axes[1].plot(test_serija.index, test_predictions, color='red', linewidth=1.5, 
             linestyle='--', label='ARIMA predikcije')
axes[1].fill_between(test_serija.index,
                     test_serija.values, test_predictions,
                     alpha=0.15, color='red', label='Greška')
axes[1].set_title(f'Test period (uvecano) | RMSE={test_rmse:.4f} | MAE={test_mae:.4f} | MAPE={test_mape:.2f}%')
axes[1].set_ylabel('USD/galon')
axes[1].set_xlabel('Datum')
axes[1].legend()
 
plt.tight_layout()
plt.savefig('results/arima_predikcije.png', dpi=150, bbox_inches='tight')
plt.show()

# COVID pad (2020-01 do 2020-07), model prati nagli pad jako dobro, cak i taj ekstremni minimum
# Oporavak (2020-07 do 2021) - rastuci trend pracen bez problema
# Stabilni periodi (2018-2019)  gotovo identicne linije

# Jedina mesta gde se vidi malo odvajanje je
# na nekim vrhovima i dolinama crvena malo kasni za plavom,
# to je tipicno za walk-forward ARIMA, model uvek predvidja jedan
# korak unapred pa blago kasni na naglim promenama pravca, ali razlika je minimalna
 
# %% residual plot
 
reziduali_test = test_serija.values - test_predictions
 
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
 
axes[0].plot(test_serija.index, reziduali_test, color='purple', linewidth=0.8)
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0].set_title('Reziduali na test skupu (stvarna - predvidjena vrednost)')
axes[0].set_ylabel('Greska (USD/galon)')
 
axes[1].hist(reziduali_test, bins=40, color='purple', alpha=0.7, edgecolor='white')
axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_title('Distribucija reziduala')
axes[1].set_xlabel('Greska (USD/galon)')
axes[1].set_ylabel('Frekvencija')
 
plt.tight_layout()
plt.savefig('results/arima_reziduali.png', dpi=150, bbox_inches='tight')
plt.show()

# Reziduali osciluju nasumicno oko nule kroz ceo period
# Nema sistematskog obrasca, nema trenda nagore ili nadole
# Nema perioda gde su reziduali konzistentno pozitivni ili negativni,
# model ne precenjuje ni ne potcenjuje 

# ovaj grafik za distribuciju nam kaze da je distribucija centrirana oko nule,
# model ne precenjuje ni ne potcenjuje 
# Vecina gresaka je izmedju -0.05 i +0.05 USD/galon
 
# %% cuvanje modela i rezultata
 
arima_metrics = {
    'model'     : f'ARIMA({BEST_P},{BEST_D},{BEST_Q})',
    'val_rmse'  : val_rmse,
    'val_mae'   : val_mae,
    'val_mape'  : val_mape,
    'test_rmse' : test_rmse,
    'test_mae'  : test_mae,
    'test_mape' : test_mape,
}
 
arima_predictions = {
    'dates'            : test_serija.index,
    'y_true'           : test_serija.values,
    'y_pred'           : test_predictions,
    'val_predictions'  : val_predictions,
    'val_dates'        : val_serija.index,
    'val_true'         : val_serija.values,
}
 
with open('data/processed/arima_model.pkl', 'wb') as f:
    pickle.dump(arima_fit, f)
 
with open('data/processed/arima_predictions.pkl', 'wb') as f:
    pickle.dump(arima_predictions, f)
 
with open('data/processed/arima_metrics.pkl', 'wb') as f:
    pickle.dump(arima_metrics, f)
 
print("\nSacuvano:")
print("data/processed/arima_model.pkl")
print("data/processed/arima_predictions.pkl")
print("data/processed/arima_metrics.pkl")
print("ARIMA TRENIRANJE ZAVRSENO")
print(f"  Model  : ARIMA({BEST_P},{BEST_D},{BEST_Q})")
print(f"  Val  RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | MAPE: {val_mape:.2f}%")
print(f"  Test RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | MAPE: {test_mape:.2f}%")
 

# %% izracunate vrednosti: 
#    Model  : ARIMA(3,1,2)
#    Val  RMSE: 0.0432 | MAE: 0.0313 | MAPE: 1.29%
 #   Test RMSE: 0.0335 | MAE: 0.0262 | MAPE: 1.07%
