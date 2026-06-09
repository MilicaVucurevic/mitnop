# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:59:33 2026

@author: danij
"""
# %% biblioteke

import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# %% ucitavanje podataka

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('../data/processed/df_original.csv', parse_dates=['date'])
df_scaled = pd.read_csv('../data/processed/df_scaled.csv', parse_dates=['date'])

# %% vizualizacija vremenskih serija

# sve kategorije goriva
fuel_cols = ['regular_conv', 'regular_ref', 'midgrade_conv', 'midgrade_ref', 
             'premium_conv', 'premium_ref', 'diesel']

fig, axes = plt.subplots(len(fuel_cols), 1, figsize=(14, 20), sharex=True)

for i, col in enumerate(fuel_cols):
    axes[i].plot(df['date'], df[col], linewidth=0.8)
    axes[i].set_ylabel('USD/galon', fontsize=9)
    axes[i].set_title(col, fontsize=10)
    axes[i].axvspan(pd.Timestamp('2020-03-09'), pd.Timestamp('2021-02-15'), 
                    alpha=0.2, color='red', label='COVID period')
    axes[i].axvspan(pd.Timestamp('2008-08-04'), pd.Timestamp('2009-06-01'),
                    alpha=0.2, color='orange', label='Finansijska kriza')
    if i==0:
        axes[i].legend(fontsize=8)
fig.suptitle('Maloprodajne cene goriva u SAD (1995-2021)', fontsize=13, y=1.01)
plt.tight_layout() # da se naslovi, oznake i legende na preklapaju
plt.savefig('../data/processed/vizualizacija_goriva.png', dpi=150, bbox_inches='tight') #cuvamo grafik kao png fajl
plt.show()

# cene sirove nafte i usd indexa
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

axes[0].plot(df['date'], df['crude_oil'], color='black', linewidth=0.8)
axes[0].set_ylabel('USD/barel')
axes[0].set_title('Cena sirove nafte (WTI)')
axes[0].axvspan(pd.Timestamp('2020-03-09'), pd.Timestamp('2021-02-15'),
                alpha=0.2, color='red', label='COVID period')
axes[0].axvspan(pd.Timestamp('2008-08-04'), pd.Timestamp('2009-06-01'),
                alpha=0.2, color='orange', label='Finansijska kriza')
axes[0].legend(fontsize=8)

df_usd_plot = df.dropna(subset=['usd_index'])

axes[1].plot(df_usd_plot['date'], df_usd_plot['usd_index'], color='green', linewidth=0.8)
axes[1].set_ylabel('Index')
axes[1].set_title('Kurs americkog dolara (USD index)')

plt.tight_layout()
plt.savefig('../data/processed/vizualizacija_nafta_usd.png', dpi=150, bbox_inches='tight')
plt.show()

# komentari za sve kategorije goriva i za dolar
# radi uocavanja dugorocnog trenda i sezonalnosti

# Cene goriva i nafte pokazuju jasan rastuci trend od 1995. do 2008, a zatim nestabilnost.
# Krizni periodi:
# Finansijska kriza (2008-2009) - nagli pad pa zatim oporavak
# COVID 2020 - nagli pad cena, oporavak 2021.
# Finansijska kriza 2008: dolar ojacao
# Od 2015: ponovni rast, sa naglim skokom tokom COVID-a 2020.
# jak dolar -> jeftinija nafta

#U ranom periodu (1995–2008) uočava se blagi porast amplitude sezonske komponente paralelno sa rastom trenda, 
#što ukazuje na delimično multiplikativno ponašanje. 
#Međutim, u periodu nakon 2008. amplituda ostaje stabilna, 
#stoga je aditivni STL model ocenjen kao prihvatljiv pristup za exploratornu analizu.

# %% primena STL dekompozicije

for col in fuel_cols:
    series = df.set_index('date')[col].dropna()
    
    stl = STL(series, period=52, robust=True) # period je 52, jer godina ima 52 nedelje, a podaci su nedeljni
    result = stl.fit()                        # robust=True smanjuje uticaj autlajera
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    axes[0].plot(series, linewidth=0.8)
    axes[0].set_title('Originalna serija')
    axes[0].set_ylabel('USD/galon')
    
    axes[1].plot(result.trend, linewidth=0.8, color='orange')
    axes[1].set_title('Trend')
    axes[1].set_ylabel('USD/galon')
    
    axes[2].plot(result.seasonal, linewidth=0.8, color='green')
    axes[2].set_title('Sezonalnost')
    axes[2].set_ylabel('USD/galon')
    
    axes[3].plot(result.resid, linewidth=0.8, color='red')
    axes[3].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[3].set_title('Rezidual')
    axes[3].set_ylabel('USD/galon')
    
    fig.suptitle(f'STL dekompozicija - {col}', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'../data/processed/stl_{col}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
# %% analiza korelacije izmedju cene goriva, sirove nafte i USD indeksa

#heatmap korelacione matrice

# koristimo period od 2006 kad imamo sve tri varijable
df_corr = df[df['date'] >= '2006-01-01'].copy()

cols_corr = ['regular_conv', 'regular_ref', 'midgrade_conv', 'midgrade_ref',
             'premium_conv', 'premium_ref', 'diesel', 'crude_oil', 'usd_index']

corr_matrix = df_corr[cols_corr].corr()

fig, ax = plt.subplots(figsize=(10, 8)) # sa ax je kod citljiviji
sns.heatmap(corr_matrix,                # i lakse se kontrolise svaki podgrafik ako postoje
            annot=True,        
            fmt='.2f',         
            cmap='coolwarm',   # plava=negativna, crvena=pozitivna korelacija
            vmin=-1, vmax=1,
            ax=ax)

ax.set_title('Korelaciona matrica - cene goriva, nafta i USD index (2006-2021)', fontsize=12)
plt.tight_layout()
plt.savefig('../data/processed/heatmap_korelacija.png', dpi=150, bbox_inches='tight')
plt.show()
 
# Sa korelacione matrice mozemo videti sledece:
# Sve kategorije goriva su veoma jako pozitivno korelacione medjusobno (0.91-0.99)
# Jaka pozitivna korelacija izmedju goriva i sirove nafte (0.76-0.92)
# USD index - negativna korelacija sa svim varijablama
# Najjaca negativna korelacija je izmedju crude_oil i usd_index

# Visoka korelacija crude_oil sa gorivima opravdava ukljucivanje cene nafte kao ulazne varijable za LSTM
# Negativna korelacija usd_index opravdava ukljucivanje USD indeksa
# Sva goriva se krecu gotovo identicno pa je dovoljno modelovati samo jednu kategoriju (regular_conv)
# Veoma visoka korelacija izmedju gorivima znaci da ne bi imalo smisla ubaciti vise kategorija goriva istovremeno
# kao ulazne varijable u LSTM - davale bi iste informacije


# %% box-plot dijagram po godinama radi analize varijabilnosti
# dodajemo kolonu godina za grupisanje
df['year'] = df['date'].dt.year

# box-plot za regular_conv po godinama
fig, ax = plt.subplots(figsize=(22, 6))

years = df['year'].unique()
data_by_year = [df[df['year'] == y]['regular_conv'].dropna().values for y in years]

ax.boxplot(data_by_year, labels=years)
ax.set_title('Varijabilnost cene regular_conv po godinama (1995-2021)', fontsize=12)
ax.set_xlabel('Godina')
ax.set_ylabel('USD/galon')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('../data/processed/boxplot_regular_conv.png', dpi=150, bbox_inches='tight')
plt.show()

# koristimo samo regular_conv kao predstavnika svih kategorija goriva
# zbog prethodno utvrdjene jake pozitivne korelacije izmedju kategorija (0.91-0.99)
# isti obrazac varijabilnosti ocekuje se i kod ostalih kategorija

# citanje box-plot dijagrama
# visoka kutija = velika varijabilnost -> cene su mnogo oscilovale te godine
# niska kutija = mala varijabilnost -> cene su bile stabilne
# mnogo tackica = puno ekstremnih vrednosti te godine

# za regular_conv
# Cene regular benzina bile su stabilne i niske 1995-2003 (oko 1 USD/galon, male kutije).
# Varijabilnost raste od 2004, sa vrhuncem u 2008. godini (finansijska kriza - najveca kutija).
# Period 2011-2014: cene visoke ali relativno stabilne (kutije na visokom nivou).
# COVID 2020: mnostvo autlajera ispod - nagli pad cena, ekstremne vrednosti.
# 2021: oporavak sa rastucim trendom.

# %% adf test stacionarnosti

def adf_test(serija, naziv):
    rezultat = adfuller(serija.dropna(), autolag='AIC')
    
    print(f"\n{'='*50}")
    print(f"ADF test: {naziv}")
    print(f"{'='*50}")
    print(f"  Test statistika : {rezultat[0]:.4f}")
    print(f"  p-value         : {rezultat[1]:.4f}")
    print(f"  Broj lagova (AIC): {rezultat[2]}")
    print(f"  Broj opservacija: {rezultat[3]}")
    print("  Kriticne vrednosti:")
    for nivo, vrednost in rezultat[4].items():
        print(f"    {nivo}: {vrednost:.4f}")
    
    stacionarna = rezultat[1] < 0.05
    print(f"\n  >> {'STACIONARNA (odbacujemo H0)' if stacionarna else 'NIJE STACIONARNA (ne mozemo odbaciti H0)'}")
    return stacionarna

# testiramo na regular_conv kao reprezentativan primer svih goriva
# (korelaciona analiza pokazala da se sve kategorije krecu gotovo identicno)

serija_originalna = df.set_index('date')['regular_conv'].dropna()

print("\n>>> TEST NA ORIGINALNOJ SERIJI:")
stacionarna = adf_test(serija_originalna, "regular_conv (originalna)")

if not stacionarna:
    # prvo diferenciranje
    serija_diff1 = serija_originalna.diff().dropna()
    
    print("\n>>> TEST NAKON 1. DIFERENCIRANJA:")
    stacionarna_diff1 = adf_test(serija_diff1, "regular_conv (1. diferenciranje)")
     
    if stacionarna_diff1:
        d = 1
    else:
        # drugo diferenciranje, ukoliko je potrebno
        serija_diff2 = serija_diff1.diff().dropna()
        
        print("\n>>> TEST NAKON 2. DIFERENCIRANJA:")
        adf_test(serija_diff2, "regular_conv (2. diferenciranje)")
        d = 2
else:
    d = 0

print(f"\n{'='*50}")
print(f"ZAKLJUCAK: parametar d za ARIMA = {d}")
print(f"{'='*50}")
