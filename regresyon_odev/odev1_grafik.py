import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score

SEED = 13
p = np.linspace(-2, 2, 500).reshape(-1, 1)
y_true = 1 + np.sin(np.pi * p / 4)

def kazanan_icin_grafik_ciz(aktivasyon, ogrenme_orani, optimizer_tipi, baslangic_dagilimi, gizli_katman_sayisi, noron_sayisi, tablo_adi):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    model = Sequential()
    model.add(Input(shape=(1,)))
    model.add(Dense(noron_sayisi, activation=aktivasyon, kernel_initializer=baslangic_dagilimi))
    for _ in range(gizli_katman_sayisi - 1):
        model.add(Dense(noron_sayisi, activation=aktivasyon, kernel_initializer=baslangic_dagilimi))
    model.add(Dense(1, activation='linear', kernel_initializer=baslangic_dagilimi))
    
    if optimizer_tipi.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=ogrenme_orani)
    elif optimizer_tipi.lower() == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=ogrenme_orani)
    elif optimizer_tipi.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=ogrenme_orani)
        
    model.compile(optimizer=opt, loss='mse')
    history = model.fit(p, y_true, epochs=300, verbose=0)
    y_pred = model.predict(p, verbose=0)
    r2 = r2_score(y_true, y_pred)
    
    # Grafikleri Çizdirme
    plt.figure(figsize=(15, 4))

    # 1. Grafik: Epoch / MSE Error (log)
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], color='red')
    plt.yscale('log')
    plt.title("Her epoch'taki kayıp (logaritmik eksen)")
    plt.xlabel("epoch")
    plt.ylabel("loss L = 1/2(t - a²)²")
    plt.grid(True, alpha=0.3)

    # 2. Grafik: Epoch / Hedef - Çıkış Grafiği
    plt.subplot(1, 3, 2)
    plt.plot(p, y_true, label='Hedef', color='#1f497d', linewidth=2)
    plt.plot(p, y_pred, label='Ağ çıkışı', color='#c0504d', linewidth=2)
    plt.title("Hedef ve Ağ Çıkışı")
    plt.xlabel("p (giriş)")
    plt.ylabel("a² (çıkış)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Grafik: R2 Grafiği
    plt.subplot(1, 3, 3)
    plt.scatter(y_true, y_pred, color='purple', alpha=0.5, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
    plt.title(f"R2 Grafiği (R2 = {r2:.4f})")
    plt.xlabel("Gerçek Değerler (Hedef)")
    plt.ylabel("Tahmin Edilen Değerler")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # Ekrana göstermek yerine DOSYAYA KAYDET!
    dosya_adi = f"{tablo_adi}_Grafik.png"
    plt.savefig(dosya_adi, dpi=300)
    plt.close()
    print(f"BİTTİ! Grafik '{dosya_adi}' kaydedildi.")

# --- BÜTÜN TABLOLARIN KAZANANLARI İÇİN TEK TEK GRAFİK ÇİZDİRİYORUZ ---

print("Grafikler oluşturuluyor, lütfen bekleyin...")

# Tablo 1 Kazananı: Hiperbolik Tanjant
kazanan_icin_grafik_ciz(aktivasyon='tanh', ogrenme_orani=0.1, optimizer_tipi='sgd', baslangic_dagilimi='normal', gizli_katman_sayisi=1, noron_sayisi=2, tablo_adi="Tablo1")

# Tablo 2 Kazananı: Öğrenme Oranı 0.01
kazanan_icin_grafik_ciz(aktivasyon='relu', ogrenme_orani=0.01, optimizer_tipi='sgd', baslangic_dagilimi='normal', gizli_katman_sayisi=1, noron_sayisi=2, tablo_adi="Tablo2")

# Tablo 3 Kazananı: Optimizasyon SGD
kazanan_icin_grafik_ciz(aktivasyon='relu', ogrenme_orani=0.1, optimizer_tipi='sgd', baslangic_dagilimi='normal', gizli_katman_sayisi=1, noron_sayisi=2, tablo_adi="Tablo3")

# Tablo 4 Kazananı: Dağılım Uniform
kazanan_icin_grafik_ciz(aktivasyon='relu', ogrenme_orani=0.1, optimizer_tipi='sgd', baslangic_dagilimi='uniform', gizli_katman_sayisi=1, noron_sayisi=2, tablo_adi="Tablo4")

# Tablo 5 Kazananı: 1 Katman (Temel ayarlardaki ile aynı aslında)
kazanan_icin_grafik_ciz(aktivasyon='relu', ogrenme_orani=0.1, optimizer_tipi='sgd', baslangic_dagilimi='normal', gizli_katman_sayisi=1, noron_sayisi=2, tablo_adi="Tablo5")

# Tablo 6 Kazananı: 20 Nöron
kazanan_icin_grafik_ciz(aktivasyon='relu', ogrenme_orani=0.1, optimizer_tipi='sgd', baslangic_dagilimi='normal', gizli_katman_sayisi=1, noron_sayisi=20, tablo_adi="Tablo6")

print("Tüm grafikler başarıyla oluşturuldu ve sol taraftaki klasöre kaydedildi!")