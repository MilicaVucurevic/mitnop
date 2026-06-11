# Predikcija maloprodajnih cena goriva u SAD - PetroVision

Komparativna analiza vremenskih serija i dubokog učenja za predikciju nedeljnih maloprodajnih cena regularnog benzina u SAD. Projekat poredi šest modela - ARIMA, ARIMAX, LSTM, GRU i njihove varijante trenirane na uravnoteženom skupu podataka - uz Diebold-Mariano test za statističku validaciju rezultata.

# Podaci

- Weekly Retail Gasoline and Diesel Prices (EIA) | 1995-2021 | nedeljno |
- Cushing OK WTI Spot Price FOB (EIA) | 1995-2021 | nedeljno |
- USD Index (FRED) | 2006-2021 | dnevno -> agregirano na nedeljno |

Sirovi podaci se čuvaju u 'data/raw/', obradjeni u 'data/processed/'.

# Struktura projekta

projekat/
|
|-data/
|   |- raw/                         # Sirovi CSV fajlovi
|   |- processed/                   # Obrađeni podaci i modeli (.csv, .pkl, .keras)
|
|-code/
|   |-treniranje/
|       |- 04_arima.py               
|       |- arimax.py               
|       |- 04_lstm.py                       
|       |- 04_gru.py                      
|       |- lstm_mixed.py                   
|       |- gru_mixed.py            
|   |- 01_ucitavanje_podataka.py    
|   |- 02_eda.py                       
|   |- 03_feature_engineering.py       
|   |- feature_engineering_mixed.py   
|   |- 05_dm_test.py
|   |- walkForward_LSTM_GRU.ipynb                   
|
|- results/                         # Grafici (PNG)
|- requirements.txt
|- README.md
|- Izvestaj.pdf

# Pokretanje

### 1. Instalacija

pip install -r requirements.txt

### 2. Priprema podataka

python 01_ucitavanje_podataka.py
python 02_eda.py
python 03_feature_engineering.py
python feature_engineering_mixed.py

### 3. Treniranje modela

python 04_arima.py
python arimax.py
python 04_lstm.py
python 04_gru.py
python lstm_mixed.py
python gru_mixed.py

#### Walk-forward validacija za LSTM/GRU (Google Colab):
Za pokretanje Walk-forward validacije koristi se sveska walkForward_LSTM_GRU.ipynb. Zbog kompleksnosti modela i više foldova, preporučuje se pokretanje na Google Colab platformi uz korišćenje GPU ubrzanja.
Uputstvo za pokretanje:
    1.Potrebno je ubaciti folder sa projektom na svoj Google Drive
    2.Potrebno je povezati Google Colab sa svojim drajvom, otvoriti walkForward_LSTM_GRU.ipynb unutar Google Colab okruženja
    3.Kliknuti na Runtime, pa zatim na Change Runtime Type, zatim je neophodno izabrati pod hardware accelerator T4 GPU i kliknuti Save
    4.Pokrenuti prvu ćeliju u svesci koja montira Drive i proverava da li je GPU uspešno dodeljen
    5.Unutar sveske proveriti i po potrebi prilagoditi promenljivu BASE_PATH koja bi pokazivala na tačnu lokaciju foldera na Vašem Google Drive-u
    6.Izabrati Runtime Run all za izvršavanje kompletne validacije. Rezultujući grafikoni biće automatski sačuvani na naznačenoj putanji

### 4. Statističko poređenje

python 05_dm_test.py

Fajlovi se moraju pokretati redom. Svaki korak generiše međurezultate koji su ulaz za sledeći.

# Metodologija

Ciljna varijabla: nedeljna maloprodajna cena regularnog benzina ('regular_conv', USD/galon)

Podela podataka: hronološka podela 70% / 15% / 15% (train / val / test)

Modeli:

- ARIMA(3,1,2) - univarijantni, period 1995-2021, walk-forward validacija
- ARIMAX(1,1,1) - multivarijantni (+ nafta, + dolar), period 2006-2021, walk-forward validacija
- LSTM / GRU - arhitektura sa dva sloja (64,32 neurona), Dropout, Early Stopping
- LSTM_mixed / GRU_mixed - isti modeli trenirani na uravnoteženom skupu (70% iz svakog perioda: normalan, finansijska kriza, COVID)

Feature engineering (LSTM/GRU): lagged features za lagove 1, 4 i 8 nedelja, rolling mean i std za prozore od 4 i 12 nedelja, sliding window look-back = 4.

# Autori

Milica Vučurević IN 36/2023
Danijela Mijić IN 59/2023

# Napomene
Pokretati ćeliju po ćeliju u Spyder 6 okruženju, zbog walk-forward validacije u fajlovima arimax.py i 04_arima.py koja se izvršava malo duže.
