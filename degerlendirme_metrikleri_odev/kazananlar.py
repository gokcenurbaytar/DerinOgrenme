import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from sklearn.metrics import mean_squared_error, r2_score

SEED = 13
p = np.linspace(-2, 2, 500).reshape(-1, 1)

# Hocanın düzelttiği hedef fonksiyonu: y = 1 + sin(pi + p / 4)
y_true = 1 + np.sin(np.pi + (p / 4))


def grafik_ciz_ve_kaydet(history, y_pred, mse, rmse, r2, baslik, dosya_adi):
    plt.figure(figsize=(15, 4))
    plt.suptitle(f"{baslik}  |  MSE: {mse:.6f}  |  RMSE: {rmse:.4f}  |  R2: {r2:.4f}", fontsize=12, fontweight='bold')

    # 1. Grafik: Epoch / MSE Error (log)
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], color='#c0392b', linewidth=1.8)
    plt.yscale('log')
    plt.title("Her epoch'taki kayıp (logaritmik eksen)")
    plt.xlabel("epoch")
    plt.ylabel("loss L = 1/2(t - a²)²")
    plt.grid(True, which="both", alpha=0.3)

    # 2. Grafik: Hedef ve Ağ Çıkışı
    plt.subplot(1, 3, 2)
    plt.plot(p, y_true, label='Hedef', color='#1f497d', linewidth=2.5)
    plt.plot(p, y_pred, label='Ağ çıkışı', color='#c0504d', linewidth=2.0)
    plt.title("Hedef ve Ağ Çıkışı")
    plt.xlabel("p (giriş)")
    plt.ylabel("a² (çıkış)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Grafik: R2 Grafiği
    plt.subplot(1, 3, 3)
    plt.scatter(y_true, y_pred, color='purple', alpha=0.5, s=12)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=1.8)
    plt.title(f"R2 Grafiği (R2 = {r2:.4f})")
    plt.xlabel("Gerçek Değerler (Hedef)")
    plt.ylabel("Tahmin Edilen Değerler")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(dosya_adi, dpi=300)
    plt.close()


def baslatici_getir(init_adi, seed_degeri):
    if init_adi == 'uniform':
        return tf.keras.initializers.GlorotUniform(seed=seed_degeri)
    return tf.keras.initializers.GlorotNormal(seed=seed_degeri)


# Sabit temel ayarlar: ReLU, lr=0.1, SGD, normal, 1 katman, 2 nöron
def deney_calistir(aktivasyon='relu', lr=0.1, opt_adi='SGD', init='normal', katman=1, noron=2):
    tf.keras.backend.clear_session()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    model = Sequential()
    model.add(Input(shape=(1,)))

    for i in range(katman):
        k_init = baslatici_getir(init, SEED + i)
        b_init = tf.keras.initializers.Constant(0.5)

        if aktivasyon == 'leaky_relu':
            model.add(Dense(noron, kernel_initializer=k_init, bias_initializer=b_init))
            model.add(LeakyReLU())
        else:
            model.add(Dense(noron, activation=aktivasyon, kernel_initializer=k_init, bias_initializer=b_init))

    out_init = baslatici_getir(init, SEED + 99)
    model.add(Dense(1, activation='linear', kernel_initializer=out_init))

    if opt_adi.upper() == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=lr, clipnorm=1.0)
    elif opt_adi.upper() == 'RMSPROP':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr, clipnorm=1.0)
    elif opt_adi.upper() == 'ADAM':
        opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    model.compile(optimizer=opt, loss='mse')

    baslangic = time.time()
    history = model.fit(p, y_true, epochs=300, batch_size=64, verbose=0)
    sure = time.time() - baslangic

    y_pred = model.predict(p, verbose=0)

    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        return 999.0, 999.0, 0.0, sure, history, np.zeros_like(y_true)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mse, rmse, r2, sure, history, y_pred


def tablo_calistir(tablo_no, baslik, deney_listesi):
    print(f"\n=== TABLO {tablo_no}: {baslik} ===", flush=True)
    print(f"{'Deney':<20} | {'MSE':<10} | {'RMSE':<8} | {'R2':<8} | {'Süre (sn)'}", flush=True)
    print("-" * 68, flush=True)

    en_iyi = None
    for ad, kwargs in deney_listesi:
        mse, rmse, r2, sure, hist, y_pred = deney_calistir(**kwargs)
        print(f"{ad:<20} | {mse:<10.6f} | {rmse:<8.4f} | {r2:<8.4f} | {sure:.2f}", flush=True)

        if en_iyi is None or mse < en_iyi["mse"]:
            en_iyi = {
                "ad": ad, "mse": mse, "rmse": rmse, "r2": r2,
                "hist": hist, "y_pred": y_pred
            }

    grafik_ciz_ve_kaydet(
        en_iyi["hist"], en_iyi["y_pred"], en_iyi["mse"], en_iyi["rmse"], en_iyi["r2"],
        f"Tablo {tablo_no} Kazananı ({en_iyi['ad']})",
        f"Tablo{tablo_no}_Grafik.png"
    )
    print(f"--> Kazanan grafik kaydedildi: Tablo{tablo_no}_Grafik.png ({en_iyi['ad']})", flush=True)


# --- 6 TABLONUN ÇALIŞTIRILMASI ---

tablo_calistir(1, "AKTİVASYON FONKSİYONLARI", [
    ("Lineer", {"aktivasyon": "linear"}),
    ("Sigmoid", {"aktivasyon": "sigmoid"}),
    ("Hiperbolik Tanjant", {"aktivasyon": "tanh"}),
    ("ReLU", {"aktivasyon": "relu"}),
    ("LeakyReLU", {"aktivasyon": "leaky_relu"}),
    ("ELU", {"aktivasyon": "elu"}),
    ("SiLU", {"aktivasyon": "silu"}),
])

tablo_calistir(2, "ÖĞRENME ORANI", [
    ("0.1", {"lr": 0.1}),
    ("0.01", {"lr": 0.01}),
    ("0.001", {"lr": 0.001}),
])

tablo_calistir(3, "OPTİMİZASYON YÖNTEMİ", [
    ("SGD", {"opt_adi": "SGD"}),
    ("RMSProp", {"opt_adi": "RMSPROP"}),
    ("ADAM", {"opt_adi": "ADAM"}),
])

tablo_calistir(4, "BAŞLANGIÇ DAĞILIMI", [
    ("Normal", {"init": "normal"}),
    ("Uniform", {"init": "uniform"}),
])

tablo_calistir(5, "GİZLİ KATMAN SAYISI", [
    ("1 Katman", {"katman": 1, "noron": 2}),
    ("2 Katman", {"katman": 2, "noron": 2}),
    ("5 Katman", {"katman": 5, "noron": 2}),
    ("10 Katman", {"katman": 10, "noron": 2}),
    ("20 Katman", {"katman": 20, "noron": 2}),
])

tablo_calistir(6, "NÖRON SAYISI", [
    ("2 Nöron", {"katman": 1, "noron": 2}),
    ("5 Nöron", {"katman": 1, "noron": 5}),
    ("10 Nöron", {"katman": 1, "noron": 10}),
    ("20 Nöron", {"katman": 1, "noron": 20}),
    ("50 Nöron", {"katman": 1, "noron": 50}),
])

print("\nTAMAMLANDI! 6 tablo hesaplandı ve 6 kazanan grafik ana klasöre kaydedildi.", flush=True)