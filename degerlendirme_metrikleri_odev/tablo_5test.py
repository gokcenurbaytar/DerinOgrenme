import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from sklearn.metrics import mean_squared_error, r2_score

SEED = 13
p = np.linspace(-2, 2, 500).reshape(-1, 1)
y_true = 1 + np.sin(np.pi + (p / 4))


def tablo5_test(mod_adi, aktivasyon, opt_adi, bias_val, init_tipi):
    print(f"\n--- {mod_adi} ---", flush=True)
    for katman in [1, 2, 5, 10, 20]:
        tf.keras.backend.clear_session()
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        model = Sequential()
        model.add(Input(shape=(1,)))
        for i in range(katman):
            if init_tipi == 'glorot':
                k_init = tf.keras.initializers.GlorotNormal(seed=SEED + i)
            else:
                k_init = tf.keras.initializers.HeNormal(seed=SEED + i)
            b_init = tf.keras.initializers.Constant(bias_val)

            if aktivasyon == 'leaky_relu':
                model.add(Dense(2, kernel_initializer=k_init, bias_initializer=b_init))
                model.add(LeakyReLU())
            else:
                model.add(Dense(2, activation=aktivasyon, kernel_initializer=k_init, bias_initializer=b_init))

        model.add(Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.GlorotNormal(seed=SEED + 99)))

        if opt_adi == 'ADAM':
            opt = tf.keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, clipnorm=1.0)

        model.compile(optimizer=opt, loss='mse')
        bas = time.time()
        model.fit(p, y_true, epochs=300, batch_size=64, verbose=0)
        sure = time.time() - bas

        y_pred = model.predict(p, verbose=0)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print(f"{katman:>2} Katman | MSE: {mse:.6f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | Süre: {sure:.2f} sn", flush=True)


# Seçenek A: ReLU + GlorotNormal + Pozitif Bias (0.5) + Momentumlu SGD
tablo5_test("SEÇENEK A (ReLU + SGD)", "relu", "SGD", 0.5, "glorot")

# Seçenek B: Önceki tabloların kazananlarıyla (LeakyReLU + ADAM)
tablo5_test("SEÇENEK B (LeakyReLU + ADAM)", "leaky_relu", "ADAM", 0.1, "he")