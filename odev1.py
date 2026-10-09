X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1] #and kapısı

# başlangıç ağırlıkları ve bias değerlerini kurala uygun olarak rastgele belirledim
w = [0.47, 0.98]
b = 0.03
# 2 Epoch boyunca hesaplama icin
w2 = [0.47, 0.98]
b2 = 0.03

ogrenme_orani = 0.1
epoch = 0
# sadece 2 kez dönecek bir for döngüsü kuruyoruz
for epoch_kontrol in range(2):
    w_degisim = [0, 0]
    b_degisim = 0
    
    for i in range(len(X)):
        denklem = w2[0] * X[i][0] + w2[1] * X[i][1] + b2
        tahmin = 1 if denklem > 0 else 0
        hata = y[i] - tahmin
        
        w_degisim[0] += ogrenme_orani * hata * X[i][0]
        w_degisim[1] += ogrenme_orani * hata * X[i][1]
        b_degisim += ogrenme_orani * hata
        
    w2[0] += w_degisim[0]
    w2[1] += w_degisim[1]
    b2 += b_degisim

print("2. epoch sonundaki ara değerler:")
print("w:", [round(w2[0], 2), round(w2[1], 2)], "b:", round(b2, 2))
print("\n")

while True:
    toplam_hata = 0
    w_degisim = [0, 0]
    b_degisim = 0
    
# batch ve epoch eşit istendiği için tüm verileri gezip hataları topluyoruz
    for i in range(len(X)):
        denklem_sonucu = w[0] * X[i][0] + w[1] * X[i][1] + b
        tahmin = 1 if denklem_sonucu > 0 else 0
        
        hata = y[i] - tahmin
        
        # değişimleri formüle göre hesaplayıp ayrı bir değişkende tutuyorum
        w_degisim[0] += ogrenme_orani * hata * X[i][0]
        w_degisim[1] += ogrenme_orani * hata * X[i][1]
        b_degisim += ogrenme_orani * hata
        toplam_hata += abs(hata)
        
    # her veri adımında değil sadece epoch bitiminde ağırlıkları tek seferde güncelliyorum
    w[0] += w_degisim[0]
    w[1] += w_degisim[1]
    b += b_degisim
    epoch += 1
    
    # toplam hata sıfırlandığında yani denklem doğru çalıştığında eğitimi bitiriyorum
    if toplam_hata == 0:
        break

    # sonuçları ekrana yazdırma 
print("epoch:", epoch)
print("w:", [round(w[0], 2), round(w[1], 2)], "b:", round(b, 2))
for i in range(len(X)):
    denklem_sonucu = w[0] * X[i][0] + w[1] * X[i][1] + b
    tahmin = 1 if denklem_sonucu > 0 else 0
    print(f"{X[i]} {y[i]} {tahmin}")