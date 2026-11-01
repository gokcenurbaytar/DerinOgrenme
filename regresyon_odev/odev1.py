import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error, r2_score
import time

# 1. ADIM: HOCANIN İSTEDİĞİ SABİT ÇEKİRDEK (OKUL NO SON 2 HANE: 13)
SEED = 13

# 2. ADIM: GİRİŞ VE HEDEF VERİSİNİ OLUŞTURMA
# p değerleri -2 ile 2 arasında
p = np.linspace(-2, 2, 500).reshape(-1, 1)
# Hedef formülü y = 1 + sin(pi * p / 4)
y_true = 1 + np.sin(np.pi * p / 4)

def deney_yap(aktivasyon='relu', ogrenme_orani=0.1, optimizer_tipi='sgd', 
              baslangic_dagilimi='normal', gizli_katman_sayisi=1, noron_sayisi=2):
    
    # Her deneyin tıpatıp aynı rastgelelikle başlaması için seed'leri sıfırlıyoruz
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    model = Sequential()
    model.add(Input(shape=(1,))) # Girdi katmanı eklendi (uyarıyı gidermek için)
    
    # İlk gizli katman
    model.add(Dense(noron_sayisi, activation=aktivasyon, kernel_initializer=baslangic_dagilimi))
    
    # Eğer hoca 2, 5, 10 gizli katman isterse (Tablo 5 için) burası çalışacak
    for _ in range(gizli_katman_sayisi - 1):
        model.add(Dense(noron_sayisi, activation=aktivasyon, kernel_initializer=baslangic_dagilimi))
        
    # Çıkış katmanı (Her zaman 1 lineer nöron)
    model.add(Dense(1, activation='linear', kernel_initializer=baslangic_dagilimi))
    
    # Optimizasyon algoritmasını seçme (Tablo 3 için)
    if optimizer_tipi.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=ogrenme_orani)
    elif optimizer_tipi.lower() == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=ogrenme_orani)
    elif optimizer_tipi.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=ogrenme_orani)
        
    model.compile(optimizer=opt, loss='mse')
    
    # Süreyi başlat ve tam 300 epoch eğit (Ekrana binlerce satır basmasın diye verbose=0)
    baslangic_suresi = time.time()
    history = model.fit(p, y_true, epochs=300, verbose=0)
    bitis_suresi = time.time()
    
    sure = bitis_suresi - baslangic_suresi
    
    # Ağa tahmin yaptır
    y_pred = model.predict(p, verbose=0)
    
    # Değerlendirme kriterleri (RMSE ve R2)
    # Hoca değerlendirme kriteri olarak RMSE, MSE grafiği ve tablolarda MSE istemiş. 
    # Excel'deki tablo başlığı "MSE" olduğu için MSE hesaplıyoruz.
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return history, y_pred, mse, r2, sure

# --- TABLO 1: AKTİVASYON FONKSİYONLARININ DEĞİŞİMİ ---
print("\n=== TABLO 1: AKTİVASYON FONKSİYONLARI DENEYLERİ ===")
print("(Sabitler: LR=0.1, Opt=SGD, Init=normal, Katman=1, Nöron=2)")
print("-" * 60)
print(f"{'Aktivasyon':<20} | {'MSE':<10} | {'R2 Skoru':<10} | {'Süre (sn)':<10}")
print("-" * 60)

# Keras'taki tam isimleriyle aktivasyon fonksiyonları
aktivasyonlar = ['linear', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'swish']
gorunen_isimler = ['Lineer', 'Sigmoid', 'Hiperbolik Tanjant', 'ReLU', 'LeakyReLU', 'ELU', 'SiLU']

for act, isim in zip(aktivasyonlar, gorunen_isimler):
    history, y_pred, mse, r2, sure = deney_yap(
        aktivasyon=act, 
        ogrenme_orani=0.1, 
        optimizer_tipi='sgd', 
        baslangic_dagilimi='normal', 
        gizli_katman_sayisi=1, 
        noron_sayisi=2
    )
    print(f"{isim:<20} | {mse:<10.4f} | {r2:<10.4f} | {sure:<10.2f}")
print("-" * 60)

# --- TABLO 2: ÖĞRENME ORANI DEĞİŞİMİ ---
print("\n=== TABLO 2: ÖĞRENME ORANI DENEYLERİ ===")
print("(Sabitler: Akt=ReLU, Opt=SGD, Init=normal, Katman=1, Nöron=2)")
print("-" * 60)
print(f"{'Öğrenme Oranı':<15} | {'MSE':<10} | {'R2 Skoru':<10} | {'Süre (sn)':<10}")
print("-" * 60)

ogrenme_oranlari = [0.1, 0.01, 0.001]

for lr in ogrenme_oranlari:
    history, y_pred, mse, r2, sure = deney_yap(
        aktivasyon='relu', 
        ogrenme_orani=lr, 
        optimizer_tipi='sgd', 
        baslangic_dagilimi='normal', 
        gizli_katman_sayisi=1, 
        noron_sayisi=2
    )
    print(f"{lr:<15} | {mse:<10.4f} | {r2:<10.4f} | {sure:<10.2f}")
print("-" * 60)

# --- TABLO 3: OPTİMİZASYON YÖNTEMİ DEĞİŞİMİ ---
print("\n=== TABLO 3: OPTİMİZASYON YÖNTEMİ DENEYLERİ ===")
print("(Sabitler: Akt=ReLU, LR=0.1, Init=normal, Katman=1, Nöron=2)")
print("-" * 60)
print(f"{'Optimizasyon':<15} | {'MSE':<10} | {'R2 Skoru':<10} | {'Süre (sn)':<10}")
print("-" * 60)

optimizasyon_yontemleri = ['sgd', 'rmsprop', 'adam']

for opt in optimizasyon_yontemleri:
    history, y_pred, mse, r2, sure = deney_yap(
        aktivasyon='relu', 
        ogrenme_orani=0.1, 
        optimizer_tipi=opt, 
        baslangic_dagilimi='normal', 
        gizli_katman_sayisi=1, 
        noron_sayisi=2
    )
    # Hoca çıktıları büyük harfle istiyorsa upper() ile yazdırıyoruz
    print(f"{opt.upper():<15} | {mse:<10.4f} | {r2:<10.4f} | {sure:<10.2f}")
print("-" * 60)

# --- TABLO 4: BAŞLANGIÇ DAĞILIMI DEĞİŞİMİ ---
print("\n=== TABLO 4: BAŞLANGIÇ DAĞILIMI DENEYLERİ ===")
print("(Sabitler: Akt=ReLU, LR=0.1, Opt=SGD, Katman=1, Nöron=2)")
print("-" * 60)
print(f"{'Dağılım Tipi':<15} | {'MSE':<10} | {'R2 Skoru':<10} | {'Süre (sn)':<10}")
print("-" * 60)

dagilim_tipleri = ['normal', 'uniform']

for dagilim in dagilim_tipleri:
    history, y_pred, mse, r2, sure = deney_yap(
        aktivasyon='relu', 
        ogrenme_orani=0.1, 
        optimizer_tipi='sgd', 
        baslangic_dagilimi=dagilim, 
        gizli_katman_sayisi=1, 
        noron_sayisi=2
    )
    # Hoca Türkçe istediği için isimleri çeviriyoruz
    gorunen_isim = "Normal" if dagilim == 'normal' else "Uniform"
    print(f"{gorunen_isim:<15} | {mse:<10.4f} | {r2:<10.4f} | {sure:<10.2f}")
print("-" * 60)
# --- TABLO 5: GİZLİ KATMAN SAYISI DEĞİŞİMİ ---
print("\n=== TABLO 5: GİZLİ KATMAN SAYISI DENEYLERİ ===")
print("(Sabitler: Akt=ReLU, LR=0.1, Opt=SGD, Init=normal, Nöron=2)")
print("-" * 60)
print(f"{'Katman Sayısı':<15} | {'MSE':<10} | {'R2 Skoru':<10} | {'Süre (sn)':<10}")
print("-" * 60)

katman_sayilari = [1, 2, 5, 10, 20]

for katman in katman_sayilari:
    history, y_pred, mse, r2, sure = deney_yap(
        aktivasyon='relu', 
        ogrenme_orani=0.1, 
        optimizer_tipi='sgd', 
        baslangic_dagilimi='normal', 
        gizli_katman_sayisi=katman, 
        noron_sayisi=2
    )
    print(f"{katman:<15} | {mse:<10.4f} | {r2:<10.4f} | {sure:<10.2f}")
print("-" * 60)
# --- TABLO 6: GİZLİ KATMANDAKİ NÖRON SAYISI DEĞİŞİMİ ---
print("\n=== TABLO 6: NÖRON SAYISI DENEYLERİ ===")
print("(Sabitler: Akt=ReLU, LR=0.1, Opt=SGD, Init=normal, Katman=1)")
print("-" * 60)
print(f"{'Nöron Sayısı':<15} | {'MSE':<10} | {'R2 Skoru':<10} | {'Süre (sn)':<10}")
print("-" * 60)

noron_sayilari = [2, 5, 10, 20, 50]

for noron in noron_sayilari:
    history, y_pred, mse, r2, sure = deney_yap(
        aktivasyon='relu', 
        ogrenme_orani=0.1, 
        optimizer_tipi='sgd', 
        baslangic_dagilimi='normal', 
        gizli_katman_sayisi=1, 
        noron_sayisi=noron
    )
    print(f"{noron:<15} | {mse:<10.4f} | {r2:<10.4f} | {sure:<10.2f}")
print("-" * 60)