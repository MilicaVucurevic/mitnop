# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:44:19 2026

@author: Milica
"""

# %% dokumentacija

# Opis: Ucitavanje, ciscenje i priprema tri skupa podataka:
#       - Maloprodajne cene goriva u SAD (1995-2021)
#       - Globalne cene sirove nafte (1995-2021)  
#       - Kurs americkog dolara (2006-2021)
#
# Output: 
#       - data/processed/df_original.csv
#       - data/processed/df_scaled.csv
#       - data/processed/scaler.pkl

import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import pickle

os.chdir('C:/Users/danij/OneDrive/Desktop/treca/mitnop/projekat')
#os.chdir('C:/Users/Milica/Documents/PetroVision')

# %% cene goriva
df_gorivo = pd.read_csv('data/raw/Weekly_Retail_Gasoline_and_Diesel_Prices.csv', 
                        skiprows=6,
                        parse_dates=['Week of'])

df_gorivo.columns = ['date', 'diesel', 'midgrade_ref', 'premium_conv', 
                     'premium_ref', 'midgrade_conv', 'regular_ref', 'regular_conv']

df_gorivo = df_gorivo.sort_values('date').reset_index(drop=True)
df_gorivo = df_gorivo[(df_gorivo['date'] >= '1995-01-01') & (df_gorivo['date'] <= '2021-12-31')]

print("GORIVO:")
print(df_gorivo.shape)
print(df_gorivo.head())

# %% sirova nafta

df_nafta = pd.read_csv('data/raw/Weekly_Cushing_OK_WTI_Spot_Price_FOB.csv',
                       skiprows=4)

df_nafta.columns = ['date', 'crude_oil']
df_nafta['date'] = pd.to_datetime(df_nafta['date'])

df_nafta = df_nafta.sort_values('date').reset_index(drop=True)
df_nafta = df_nafta[(df_nafta['date'] >= '1995-01-01') & (df_nafta['date'] <= '2021-12-31')]

print("NAFTA:")
print(df_nafta.shape)
print(df_nafta.head())

# %% kurs americkog dolara

df_usd = pd.read_csv('data/raw/USD_index.csv',
                     parse_dates=['observation_date'])

df_usd.columns = ['date', 'usd_index']

# fajl je dnevni, trebamo agregirati na nedeljni nivo
df_usd = df_usd.set_index('date')
df_usd = df_usd.resample('W-SUN').mean()    # kraj nedelje je nedelja
df_usd = df_usd.reset_index()

df_usd = df_usd[(df_usd['date'] >= '2006-01-01') & (df_usd['date'] <= '2021-12-31')]

print("USD:")
print(df_usd.shape)
print(df_usd.head())

# %% spajanje po datumu

df_gorivo['date'] = df_gorivo['date'] - pd.to_timedelta(df_gorivo['date'].dt.dayofweek, unit='D')
df_nafta['date'] = df_nafta['date'] - pd.to_timedelta(df_nafta['date'].dt.dayofweek, unit='D')
df_usd['date'] = df_usd['date'] - pd.to_timedelta(df_usd['date'].dt.dayofweek, unit='D')

df = pd.merge(df_gorivo, df_nafta, on='date', how='inner')
df = pd.merge(df, df_usd, on='date', how='left')

print("SPOJENI DATAFRAME:")
print(df.shape)
print(df.head())
print(df.tail())



# %% provera nedostajucih vrednosti

print()
print("MISSING VALUES PO KOLONAMA:")
print(df.isnull().sum())
print()

# procenat nedostajucih vrednosti
print()
print("PROCENAT MISSING (%):")
print((df.isnull().sum() / len(df) * 100).round(2))
print()

# prikaz redova gde usd_index ima NaN
print()
print("REDOVI SA NaN u usd_index:")
print(f"Ukupno: {df['usd_index'].isnull().sum()} redova")
print(f"Period: {df[df['usd_index'].isnull()]['date'].min()} do {df[df['usd_index'].isnull()]['date'].max()}")

# provera da li ima NaN u usd_index posle 2006?
print()
nan_posle_2006 = df[(df['date'] >= '2006-01-01') & (df['usd_index'].isnull())]
print(f"NaN u usd_index posle 2006: {len(nan_posle_2006)} redova")
if len(nan_posle_2006) > 0:
    print(nan_posle_2006[['date', 'usd_index']].head(10))
    
# koliko NaN je pre 2006
print()
nan_pre_2006 = df[(df['date'] < '2006-01-01') & (df['usd_index'].isnull())]
print(f"NaN u usd_index pre 2006: {len(nan_pre_2006)} redova")

# poslednja nedelja u data set-u je NaN pa je fillujemo
# popunjava samo taj jedan NaN na kraju sa prethodnom vrednoscu
df['usd_index'] = df['usd_index'].ffill()

# verifikacija
print(df[df['date'] >= '2021-12-20'][['date', 'usd_index']])

# %% provera tipova kolona, da li su sve float64

print()
print("TIPOVI KOLONA:")
print(df.dtypes)
print()
print("OSNOVNE STATISTIKE:")
print(df.describe())

# %% detekcija COVID outlier-a

# pregled cena 2019-2021 da vidimo kad su se stabilizovale, da bismo znali koji tacno opseg da posmatramo
covid_period = df[(df['date'] >= '2020-03-01') & (df['date'] <= '2023-05-01')]
print()
print(covid_period[['date', 'regular_conv', 'crude_oil']].to_string())

# po podacima vidimo da u periodu pre korone, cena regular_conv bila je 2.40-2.50 usd/galonu. Pad pocinje od
# oko  2020-03-09 (2.272 benzin, 32.39 nafta, tad je bas nagli pad nafte, ovo pre bi mogle biti obicne oscilacije).
# Cene se vracaju na pred COVID nivo (znaci 2.40+) tek od 2021-02-15
# taj period od skoro godinu dana uzmemo kao COVID anomaliju. Kako ne bismo izbacivali taj period, zbog 
# nekonzistentnosti podataka (za vremenske serije i ARIMU), uvescemo flag za taj period
df['covid_flag'] = ((df['date'] >= '2020-03-09') & 
                    (df['date'] <= '2021-02-15')).astype(int)
print()
print(f"Redovi označeni kao COVID period: {df['covid_flag'].sum()}")

# %% finansijska kriza

df['crisis_flag'] = ((df['date'] >= '2008-01-01') & 
                     (df['date'] <= '2008-12-31')).astype(int)
print()
print(f"Redovi označeni kao FINANSIJSKA KRIZA: {df['crisis_flag'].sum()}")

# %% min-max normalizacija

# ovo radimo jer LSTM i GRU ne rade dobro sa sirovim podacima jer crude_oil ide do 142 USD/barelu, 
# regular_conv ide do 4.76 USD/galonu, usd_index ide do 124, posto su ove vrednosti na razlicitim skalama, 
# neuronske mreze bi davale vecu vaznost kolonama sa vecim brojevima, bukv samo jer su brojevi veci
# kad uradimo min-max normalizaciju, sve se svede na raspon [0,1] pa mreza tretira sve kolone ravnopravno

# posto ovo za arimu nije potrebno, za nju cemo koristiti originalni df, a za neuronske mreze df_scaled

# kolone koje normalizujemo (sve osim date i covid_flag)
cols_to_scale = ['diesel', 'midgrade_ref', 'premium_conv', 'premium_ref', 
                 'midgrade_conv', 'regular_ref', 'regular_conv', 
                 'crude_oil', 'usd_index']

scaler = MinMaxScaler()

# cuvamo originalni dataframe, pravimo novi sa skaliranim vrednostima
df_scaled = df.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print()
print("ORIGINALNE VREDNOSTI (prve 3 vrste):")
print(df[['date', 'regular_conv', 'crude_oil', 'usd_index']].head(3))
print()
print("SKALIRANJE VREDNOSTI (prve 3 vrste):")
print(df_scaled[['date', 'regular_conv', 'crude_oil', 'usd_index']].head(3))
print()
print("MIN/MAX po kolonama posle skaliranja:")
print(df_scaled[cols_to_scale].agg(['min', 'max']))

# %% prilagodjavanje podataka skupu USD index koji krece od 2006.

# Posto usd_index pocinje tek od 2006. godine,
# ovde odsecamo dataset na period 2006-2021 kako bismo izbegli stotine NaN vrednosti.
df_scaled = df_scaled[df_scaled['date'] >= '2006-01-01'].reset_index(drop=True)
print(f"Dimenzije nakon filtriranja (od 2006. godine): {df_scaled.shape}")
print(df_scaled.head(3))

# %% cuvanje podataka

# sacuvamo ova dataframe-a u fajlove, da ne mora svaki put da se pokrece kod
df.to_csv('data/processed/df_original.csv', index=False)
df_scaled.to_csv('data/processed/df_scaled.csv', index=False)

# Scaler cuvamo posebno u pickle fajlu jer ce trebati kasnije kada budemo radile inverznu 
# transformaciju predvidjanja nazad u stvarne vrednosti (korak 5 iz spec).
with open('data/processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Podaci sacuvani")
