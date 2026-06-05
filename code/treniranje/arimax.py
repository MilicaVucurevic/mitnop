# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:58:14 2026

@author: danij
"""

# %% dokumentacija
 
# Opis: Razvoj i treniranje ARIMAX modela za predikciju cena regular_conv benzina.
#       ARIMAX je multivarijantna metoda - koristi cene goriva, sirove nafte i USD indeksa (2006-2021).
#       Za razliku od univarijantne ARIMA koja koristi samo regular_conv (1995-2021),
#       ARIMAX ukljucuje crude_oil i usd_index kao egzogene varijable.
#       Parametar d=1 utvrđen ADF testom u 02_eda.py.
#       Parametri p i q određuju se grid search-om na osnovu AIC.
#       Walk-Forward validacija za realnu procenu generalizacije.
#
#       Kljucna razlika od univarijantne ARIMA:
#       - Period pocinje od 2006 (kada postoje podaci za USD index)
#       - Model prima egzogene prediktore u svakom koraku walk-forward petlje
#       - U walk-forward petlji koristimo stvarne vrednosti egzogenih varijabli
#         (simulira realnu upotrebu gde su podaci o nafti i dolaru dostupni)
#
# Input:
#       - data/processed/df_original.csv
#
# Output:
#       - data/processed/arimax_model.pkl
#       - data/processed/arimax_predictions.pkl
#       - data/processed/arimax_metrics.pkl
 
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
 
# ARIMAX koristi originalne vrednosti - od 2006. zbog USD indexa
df = pd.read_csv('../../data/processed/df_original.csv', parse_dates=['date'])
df = df.set_index('date')
 
# filtriramo od 2006. jer od tada imamo sve tri varijable
df = df[df.index >= '2006-01-01'].copy()
 
# ciljna varijabla
serija = df['regular_conv'].dropna()
 
# egzogene varijable - crude_oil i usd_index
# koristimo originalne vrednosti
exog = df[['crude_oil', 'usd_index']].copy()
 
# sinhronizujemo indekse - zadrzavamo samo redove gde imamo sve tri varijable
idx_zajednicki = serija.index.intersection(exog.dropna().index)
serija = serija.loc[idx_zajednicki]
exog   = exog.loc[idx_zajednicki]
 
print(f"Ukupno redova (od 2006): {len(serija)}")
print(f"Period: {serija.index.min().date()} -> {serija.index.max().date()}")
print(f"Min: {serija.min():.3f} | Max: {serija.max():.3f} | Mean: {serija.mean():.3f}")
print(f"\nEgzogene varijable: {exog.columns.tolist()}")
print(f"NaN u egzogenim: {exog.isnull().sum().to_dict()}")
 
# %% ACF i PACF grafici za odredjivanje p i q
 
# Gledamo diferenciranu seriju (d=1 vec utvrdjeno u 02_eda.py)
serija_diff = serija.diff().dropna()
 
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
 
plot_acf(serija_diff, lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title('ACF - regular_conv (nakon 1. diferenciranja, period 2006-2021)')
axes[0].set_xlabel('Lag (nedelje)')
 
plot_pacf(serija_diff, lags=40, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('PACF - regular_conv (nakon 1. diferenciranja, period 2006-2021)')
axes[1].set_xlabel('Lag (nedelje)')
 
plt.tight_layout()
#plt.savefig('../../data/processed/arimax_acf_pacf.png', dpi=150, bbox_inches='tight')
plt.show()
 
# %% grid search za parametre (p, q)
 
# d=1 je vec utvrdjeno ADF testom
# pretrazujemo iste vrednosti p i q kao u univarijantnoj ARIMA radi konzistentnosti
 
p_vrednosti = [1, 2, 3]
q_vrednosti = [1, 2, 3, 4, 5]
d = 1
 
print("\nPretrazivanje optimalnih parametara (p, d, q) na osnovu AIC...")
print(f"{'p':>4} {'d':>4} {'q':>4} {'AIC':>12} {'BIC':>12}")
print("-" * 40)
 
rezultati_grid = []
 
for p in p_vrednosti:
    for q in q_vrednosti:
        try:
            model_temp = ARIMA(serija, exog=exog, order=(p, d, q))
            fit_temp = model_temp.fit()
            rezultati_grid.append({
                'p': p, 'd': d, 'q': q,
                'aic': fit_temp.aic,
                'bic': fit_temp.bic
            })
            print(f"{p:>4} {d:>4} {q:>4} {fit_temp.aic:>12.2f} {fit_temp.bic:>12.2f}")
        except Exception as e:
            print(f"{p:>4} {d:>4} {q:>4}   GREŠKA: {e}")
 
# sortiramo po AIC-u i uzimamo najbolji
df_grid = pd.DataFrame(rezultati_grid).sort_values('aic')
print("\nNajboljih 5 kombinacija (sortirano po AIC):")
print(df_grid.head())
 
best = df_grid.iloc[0]
BEST_P, BEST_D, BEST_Q = int(best['p']), int(best['d']), int(best['q'])
print(f"\nOdabrani parametri: ARIMAX({BEST_P}, {BEST_D}, {BEST_Q})")
 
# %% podela na trening / validacioni / test skup
 
# hronoloska podela 70/15/15 - isti odnos kao LSTM/GRU/univarijantna ARIMA
n = len(serija)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)
 
train_serija = serija.iloc[:train_end]
val_serija   = serija.iloc[train_end:val_end]
test_serija  = serija.iloc[val_end:]
 
train_exog = exog.iloc[:train_end]
val_exog   = exog.iloc[train_end:val_end]
test_exog  = exog.iloc[val_end:]
 
print("\nPodela serije:")
print(f"  Train : {len(train_serija)} | {train_serija.index[0].date()} -> {train_serija.index[-1].date()}")
print(f"  Val   : {len(val_serija)} | {val_serija.index[0].date()} -> {val_serija.index[-1].date()}")
print(f"  Test  : {len(test_serija)} | {test_serija.index[0].date()} -> {test_serija.index[-1].date()}")
 
# %% treniranje finalnog ARIMAX modela na trening skupu
 
print(f"\nTreniranje ARIMAX({BEST_P}, {BEST_D}, {BEST_Q}) na trening skupu...")
arimax_model = ARIMA(train_serija, exog=train_exog, order=(BEST_P, BEST_D, BEST_Q))
arimax_fit = arimax_model.fit()
 
print(arimax_fit.summary())
 
# %% dijagram reziduala
 
fig = arimax_fit.plot_diagnostics(figsize=(14, 10))
fig.suptitle(f'Dijagnostika reziduala - ARIMAX({BEST_P},{BEST_D},{BEST_Q})', fontsize=13)
plt.tight_layout()
#plt.savefig('../../data/processed/arimax_dijagnostika.png', dpi=150, bbox_inches='tight')
plt.show()
 
# Ljung-Box test - da li su reziduali beli sum (bez autokorelacije)
lb_test = acorr_ljungbox(arimax_fit.resid, lags=[10, 20], return_df=True)
print("\nLjung-Box test (p > 0.05 = reziduali su beli sum, model je dobar):")
print(lb_test)

# 1. Standardizovani reziduali:
# Reziduali osciluju nasumicno oko nule kroz ceo period — nema trenda ni obrasca
# Jedan veliki skok oko 2008. — finansijska kriza

# 2. Histogram + KDE (gore desno):
# Narandzasta KDE kriva je uza i visa od zelene N(0,1),
# znaci vecina gresaka je blizu nule, model cesto gresi malo
# Reziduali su priblizno normalno rasporedjeni, 
# ali sa malo tezim repovima (zbog ekstrema kao sto je finansijska kriza)

# 3. Normal Q-Q plot (dole levo):
# Srednji deo lepo prati crvenu liniju, u normalnim uslovima reziduali su normalno rasporedjeni
# Levi rep jako odstupa (one izolovane tacke),  to su nagli padovi cena tokom
# finansijske krize 2008
# Desni rep blago odstupa gore — nagli skokovi cena

# 4. Correlogram (dole desno):
# Sve vrednosti od laga 1 pa nadalje su unutar plave zone
# model je naucio sve korisno sto je mogao, ostalo je sve nepredvidivo
 
# %% walk-forward validacija
 
# Walk-forward: model se postepeno prosiruje novim posmatranjima.
# Na svakom koraku predvidjamo jedan korak unapred, zatim dodajemo stvarnu vrednost.
# Egzogene varijable (crude_oil, usd_index) se u realnoj upotrebi mogu pratiti
# u realnom vremenu (objavljuju se nedeljno), pa je ovakvo koriscenje opravdano.
 
print("\nWalk-Forward validacija na validacionom skupu...")
 
istorija_y    = list(train_serija)
istorija_exog = train_exog.values.tolist()  # lista redova
val_predictions = []
 
for t in range(len(val_serija)):
    model_wf = ARIMA(
        istorija_y,
        exog=istorija_exog,
        order=(BEST_P, BEST_D, BEST_Q)
    )
    fit_wf = model_wf.fit()

    exog_sledeci = val_exog.iloc[[t]].values 
    predikcija = fit_wf.forecast(steps=1, exog=exog_sledeci)[0]
    val_predictions.append(predikcija)
 
    # azuriramo istoriju stvarnim vrednostima
    istorija_y.append(val_serija.iloc[t])
    istorija_exog.append(val_exog.iloc[t].values.tolist())
 
    if t % 50 == 0:
        print(f"  Korak {t+1}/{len(val_serija)}")
 
val_predictions = np.array(val_predictions)
 
val_rmse = np.sqrt(mean_squared_error(val_serija.values, val_predictions))
val_mae  = mean_absolute_error(val_serija.values, val_predictions)
val_mape = np.mean(np.abs((val_serija.values - val_predictions) / val_serija.values)) * 100
 
print("\nMetrike na VALIDACIONOM skupu:")
print(f"  RMSE : {val_rmse:.4f} ")
print(f"  MAE  : {val_mae:.4f} ")
print(f"  MAPE : {val_mape:.2f}%")

# Metrike na VALIDACIONOM skupu:
#  RMSE : 0.0427 
#  MAE  : 0.0300 
#  MAPE : 1.21%
 
# %% walk-forward predikcija na test skupu
 
print("\nWalk-Forward predikcija na test skupu...")
 
# prosirujemo istoriju validacionim skupom pre testiranja
istorija_y_test    = list(train_serija) + list(val_serija)
istorija_exog_test = train_exog.values.tolist() + val_exog.values.tolist()
test_predictions   = []
 
for t in range(len(test_serija)):
    model_wf = ARIMA(
        istorija_y_test,
        exog=istorija_exog_test,
        order=(BEST_P, BEST_D, BEST_Q)
    )
    fit_wf = model_wf.fit()
 
    exog_sledeci = test_exog.iloc[[t]].values
    predikcija = fit_wf.forecast(steps=1, exog=exog_sledeci)[0]
    test_predictions.append(predikcija)
 
    istorija_y_test.append(test_serija.iloc[t])
    istorija_exog_test.append(test_exog.iloc[t].values.tolist())
 
    if t % 50 == 0:
        print(f"  Korak {t+1}/{len(test_serija)}")
 
test_predictions = np.array(test_predictions)
 
test_rmse = np.sqrt(mean_squared_error(test_serija.values, test_predictions))
test_mae  = mean_absolute_error(test_serija.values, test_predictions)
test_mape = np.mean(np.abs((test_serija.values - test_predictions) / test_serija.values)) * 100
 
print("\nMetrike na TEST skupu:")
print(f"  RMSE : {test_rmse:.4f}")
print(f"  MAE  : {test_mae:.4f}")
print(f"  MAPE : {test_mape:.2f}%")

# Metrike na TEST skupu:
#  RMSE : 0.0343
#  MAE  : 0.0260 
#  MAPE : 1.10%

# Validacioni skup:
# RMSE 0.0427 — model prosecno gresi oko 4.3 centa po galonu
# MAE 0.0300 — greska je 3 centa po galonu
# MAPE 1.21% — greska je samo 1.21% od stvarne cene

# Test skup:
# Sve tri metrike su bolje nego na validacionom skupu
# Razlog je walk-forward, kada model tokom test perioda vidi prve COVID vrednosti,
# postepeno se adaptira i uci iz njih pre sledeceg koraka, pa postaje bolji

# Poredjenje sa univarijantnom ARIMA (1995-2021):
# Univarijantna ARIMA imala je Val RMSE 0.0432 i Test RMSE 0.0335
# ARIMAX je vrlo slican, razlika je minimalna (0.0005 na validaciji, 0.0008 na testu)
# Na validaciji ARIMAX je bolji po svim metrikama, dodavanje nafte i dolara 
# blago pomaze modelu u stabilnom periodu.
# Na testu je situacija obrnuta, ARIMA(3,1,2) je malo bolja po RMSE i MAPE, 
# dok je ARIMAX malo bolji po MAE
# Visi RMSE kod ARIMAX-a na test skupu znaci da je u nekom momentu spoljna varijabla
# poslala sum, pa je model napravio nekoliko malo vecih promasaja nego cista ARIMA.
# Ono sto je odlicno za oba modela jeste da su greske na test skupu manje nego na 
# validacionom skupu ( test MAPE je ~1.08%, a val MAPE je ~1.25%).
 
# %% vizualizacija predikcija na test skupu
 
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
 
# gornji grafik - ceo period 
axes[0].plot(serija.index, serija.values, color='steelblue', linewidth=0.8, label='Stvarne vrednosti')
axes[0].plot(test_serija.index, test_predictions, color='red', linewidth=1.2,
             linestyle='--', label='ARIMAX predikcije')
axes[0].axvline(x=test_serija.index[0], color='gray', linestyle=':', linewidth=1.5, label='Pocetak test skupa')
axes[0].set_title(f'ARIMAX({BEST_P},{BEST_D},{BEST_Q}) - Predikcije vs Stvarne vrednosti (2006-2021)')
axes[0].set_ylabel('USD/galon')
axes[0].legend()
 
# donji grafik - samo test period (uvecano)
axes[1].plot(test_serija.index, test_serija.values, color='steelblue', linewidth=1.5, label='Stvarne vrednosti')
axes[1].plot(test_serija.index, test_predictions, color='red', linewidth=1.5,
             linestyle='--', label='ARIMAX predikcije')
axes[1].fill_between(test_serija.index,
                     test_serija.values, test_predictions,
                     alpha=0.15, color='red', label='Greška')
axes[1].set_title(f'Test period (uvecano) | RMSE={test_rmse:.4f} | MAE={test_mae:.4f} | MAPE={test_mape:.2f}%')
axes[1].set_ylabel('USD/galon')
axes[1].set_xlabel('Datum')
axes[1].legend()
 
plt.tight_layout()
#plt.savefig('../../data/processed/arimax_predikcije.png', dpi=150, bbox_inches='tight')
plt.show()

# Gornji grafik (ceo period 2006-2021):
# Crvena isprekidana linija (predikcije) gotovo potpuno prekriva plavu,
#  (stvarne vrednosti) u test periodu
# Test skup pocinje oko 2019.

# Donji grafik (test period uvecano):
# Stabilan period (2019-2020): greska je jedva vidljiva
# COVID pad (2020-03): model prati nagli pad jako dobro
# rastuci trend je pracen bez problema, predikcije blago kasne na naglim promenama pravca,
# sto je tipicno za walk-forward, model uvek predvidja jedan korak unapred pa ne
# moze odmah da skoci na novu vrednost
# Kraj perioda (2021): greska je nesto veca tokom brzog rasta cena

# zakljucak: Model se vrlo dobro snasao na nevidjenim podacima,
# ukljucujuci i COVID period
# Roze oblast greske je kroz ceo period vrlo tanka, sto je konzistentno sa MAPE od 1.10%.
# %% residual plot
 
reziduali_test = test_serija.values - test_predictions
 
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
 
axes[0].plot(test_serija.index, reziduali_test, color='teal', linewidth=0.8)
axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[0].set_title('Reziduali na test skupu (stvarna - predvidjena vrednost)')
axes[0].set_ylabel('Greska (USD/galon)')
 
axes[1].hist(reziduali_test, bins=40, color='teal', alpha=0.7, edgecolor='white')
axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_title('Distribucija reziduala')
axes[1].set_xlabel('Greska (USD/galon)')
axes[1].set_ylabel('Frekvencija')
 
plt.tight_layout()
#plt.savefig('../../data/processed/arimax_reziduali.png', dpi=150, bbox_inches='tight')
plt.show()

# Gornji grafik — Reziduali kroz vreme:
# Reziduali osciluju nasumicno oko nule kroz ceo period — nema trenda nagore ili nadole, 
# nema perioda gde je model precenjivao ili potcenjivao cenu
# COVID period: vide se nesto vece oscilacije, model je malo teze pratio nagle promene
# Od 2021. nadalje: reziduali postaju malo manji i stabilniji
# model se malo bolje snalazi kako prima nove podatke

# Donji grafik — Distribucija reziduala:
# Distribucija je približno centrirana oko nule
# Vecina gresaka se nalazi izmedju -0.05 i +0.05 USD/galonu
# Distribucija nije savrseno simetricna
# To se verovatno desava tokom perioda brzog rasta cena (kraj 2020 i 2021) kada model malo kasni za naglim porastom

# zakljucak: Reziduali izgledaju solidno, nasumicni su, centrirani oko nule i relativno mali.
# Jedini blagi problem je asimetrija u desnom repu
# asimetrija dolazi iz asimetrije samih podataka, u test periodu ima vise nedelja
#  sa postepenim rastom cena nego sa padom, pa se pozitivne greske cesce ponavljaju
# i desni rep distribucije postaje duzi
 
# %% cuvanje modela i rezultata
 
arimax_metrics = {
    'model'     : f'ARIMAX({BEST_P},{BEST_D},{BEST_Q})',
    'val_rmse'  : val_rmse,
    'val_mae'   : val_mae,
    'val_mape'  : val_mape,
    'test_rmse' : test_rmse,
    'test_mae'  : test_mae,
    'test_mape' : test_mape,
}
 
arimax_predictions = {
    'dates'           : test_serija.index,
    'y_true'          : test_serija.values,
    'y_pred'          : test_predictions,
    'val_predictions' : val_predictions,
    'val_dates'       : val_serija.index,
    'val_true'        : val_serija.values,
}
os.makedirs('data/processed', exist_ok=True)  # kreira folder ako ne postoji
with open('data/processed/arimax_model.pkl', 'wb') as f:
    pickle.dump(arimax_fit, f)
 
with open('data/processed/arimax_predictions.pkl', 'wb') as f:
    pickle.dump(arimax_predictions, f)
 
with open('data/processed/arimax_metrics.pkl', 'wb') as f:
    pickle.dump(arimax_metrics, f)
 
print("\nSacuvano:")
print(" data/processed/arimax_model.pkl")
print(" data/processed/arimax_predictions.pkl")
print(" data/processed/arimax_metrics.pkl")
print("ARIMAX TRENIRANJE ZAVRSENO")
print(f"  Model  : ARIMAX({BEST_P},{BEST_D},{BEST_Q})")
print(f"  Val  RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | MAPE: {val_mape:.2f}%")
print(f"  Test RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | MAPE: {test_mape:.2f}%")

# %% izlaz

#  Model  : ARIMAX(1,1,1)
#  Val  RMSE: 0.0427 | MAE: 0.0300 | MAPE: 1.21%
#  Test RMSE: 0.0343 | MAE: 0.0260 | MAPE: 1.10%