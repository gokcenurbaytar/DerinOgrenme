import os
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

klasor_adi = "Tum_Deneyler_Grafikleri"
os.makedirs(klasor_adi, exist_ok=True)


def baslatici_getir(init_adi, seed_degeri):
    if init_adi == 'uniform':
        return tf.keras.initializers.GlorotUniform(seed=seed_degeri)
    return tf.keras.initializers.GlorotNormal(seed=seed_degeri)


def tum_deney_grafik_ciz(dosya_etiketi, aktivasyon='relu', lr=0.1, opt_adi='SGD', init='normal', katman=1, noron=2):
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
        mse, rmse, r2 = 999.0, 999.0, 0.0
        y_pred = np.zeros_like(y_true)
    else:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

    plt.figure(figsize=(15, 4))
    plt.suptitle(f"{dosya_etiketi}  |  MSE: {mse:.6f}  |  RMSE: {rmse:.4f}  |  R2: {r2:.4f}", fontsize=12, fontweight='bold')

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], color='#c0392b', linewidth=1.8)
    plt.yscale('log')
    plt.title("Her epoch'taki kayıp (logaritmik eksen)")
    plt.xlabel("epoch")
    plt.ylabel("loss L = 1/2(t - a²)²")
    plt.grid(True, which="both", alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(p, y_true, label='Hedef', color='#1f497d', linewidth=2.5)
    plt.plot(p, y_pred, label='Ağ çıkışı', color='#c0504d', linewidth=2.0)
    plt.title("Hedef ve Ağ Çıkışı")
    plt.xlabel("p (giriş)")
    plt.ylabel("a² (çıkış)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.scatter(y_true, y_pred, color='purple', alpha=0.5, s=12)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=1.8)
    plt.title(f"R2 Grafiği (R2 = {r2:.4f})")
    plt.xlabel("Gerçek Değerler (Hedef)")
    plt.ylabel("Tahmin Edilen Değerler")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    kayit_yolu = os.path.join(klasor_adi, f"{dosya_etiketi}.png")
    plt.savefig(kayit_yolu, dpi=300)
    plt.close()

    print(f"Kaydedildi: {dosya_etiketi:<25} | MSE: {mse:.6f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | Süre: {sure:.2f} sn", flush=True)


print("25 deneyin tamamı hesaplanıyor ve grafikler kaydediliyor...\n", flush=True)

for ad, kod in [("Lineer", "linear"), ("Sigmoid", "sigmoid"), ("Tanh", "tanh"),
                ("ReLU", "relu"), ("LeakyReLU", "leaky_relu"), ("ELU", "elu"), ("SiLU", "silu")]:
    tum_deney_grafik_ciz(f"Tablo1_Akt_{ad}", aktivasyon=kod)

for lr in [0.1, 0.01, 0.001]:
    tum_deney_grafik_ciz(f"Tablo2_LR_{lr}", lr=lr)

for opt in ['SGD', 'RMSPROP', 'ADAM']:
    tum_deney_grafik_ciz(f"Tablo3_Opt_{opt}", opt_adi=opt)

for init in ['normal', 'uniform']:
    tum_deney_grafik_ciz(f"Tablo4_Init_{init.capitalize()}", init=init)

for katman in [1, 2, 5, 10, 20]:
    tum_deney_grafik_ciz(f"Tablo5_Katman_{katman}", katman=katman, noron=2)

for noron in [2, 5, 10, 20, 50]:
    tum_deney_grafik_ciz(f"Tablo6_Noron_{noron}", katman=1, noron=noron)

print(f"\nBİTTİ! 25 deneyin tamamı '{klasor_adi}' klasörüne kaydedildi.", flush=True)