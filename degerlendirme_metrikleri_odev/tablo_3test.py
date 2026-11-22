import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error, r2_score

SEED = 13
p = np.linspace(-2, 2, 500).reshape(-1, 1)
y_true = 1 + np.sin(np.pi + (p / 4))


def tablo3_lr_test(lr_degeri):
    print(f"\n=== TABLO 3 TEST (Öğrenme Oranı = {lr_degeri}) ===", flush=True)
    print(f"{'Optimizatör':<12} | {'MSE':<10} | {'RMSE':<8} | {'R2':<8} | {'Süre (sn)'}", flush=True)
    print("-" * 58, flush=True)

    for opt_adi in ['SGD', 'RMSProp', 'ADAM']:
        tf.keras.backend.clear_session()
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        model = Sequential([
            Input(shape=(1,)),
            Dense(
                2,
                activation='relu',
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=SEED),
                bias_initializer=tf.keras.initializers.Constant(0.5)
            ),
            Dense(
                1,
                activation='linear',
                kernel_initializer=tf.keras.initializers.GlorotNormal(seed=SEED + 99)
            )
        ])

        if opt_adi == 'SGD':
            opt = tf.keras.optimizers.SGD(learning_rate=lr_degeri, clipnorm=1.0)
        elif opt_adi == 'RMSProp':
            opt = tf.keras.optimizers.RMSprop(learning_rate=lr_degeri, clipnorm=1.0)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=lr_degeri, clipnorm=1.0)

        model.compile(optimizer=opt, loss='mse')

        bas = time.time()
        model.fit(p, y_true, epochs=300, batch_size=64, verbose=0)
        sure = time.time() - bas

        y_pred = model.predict(p, verbose=0)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print(f"{opt_adi:<12} | {mse:<10.6f} | {rmse:<8.4f} | {r2:<8.4f} | {sure:.2f}", flush=True)


for lr in [0.1, 0.01, 0.001]:
    tablo3_lr_test(lr)