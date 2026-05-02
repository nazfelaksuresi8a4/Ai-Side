🔢 Temel NumPy Fonksiyonları
1) Toplama işlemleri
np.sum()
Tüm elemanların toplamı → toplam örnek sayısı
Axis ile:
np.sum(cm, axis=0)  # sütun toplamları (predicted)
np.sum(cm, axis=1)  # satır toplamları (actual)
2) Diagonal (doğru tahminler)
np.diag(cm)
True Positive’leri verir
Accuracy, recall vs. için kritik
3) Trace (diagonal toplamı)
np.trace(cm)
Tüm doğru tahminlerin toplamı
Direkt accuracy için kullanılır
4) Bölme (oran hesaplama)
np.divide()
Precision, recall gibi oranlar için
Güvenli kullanım:
np.divide(a, b, where=b!=0)
5) NaN temizleme
np.nan_to_num()
0’a bölmeden gelen NaN’leri temizler
6) Ortalama alma
np.mean()
Macro average için
7) Eleman bazlı işlemler
np.where()
Koşullu hesaplama (örneğin 0’a bölme kontrolü)
8) Veri tipi dönüşümü
cm.astype(float)
Bölme işlemlerinde integer hatasını önler
📊 Confusion Matrix’ten Metrik Formülleri

Diyelim:

cm = np.array(...)
🎯 Accuracy
accuracy = np.trace(cm) / np.sum(cm)
🎯 Precision (her sınıf için)
precision = np.diag(cm) / np.sum(cm, axis=0)
🎯 Recall
recall = np.diag(cm) / np.sum(cm, axis=1)
🎯 F1 Score
f1 = 2 * (precision * recall) / (precision + recall)
🎯 Macro Average
macro_f1 = np.mean(f1)
🧠 Mini püf noktası
axis=0 → model ne tahmin etti
axis=1 → gerçek neydi
