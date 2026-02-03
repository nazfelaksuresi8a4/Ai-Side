import numpy as np 
import matplotlib.pyplot as  plt 

x = np.array([[6000],[8200],[9000],[14200],[16200]]).reshape(-1,1)
y = [86000,82000,78000,75000,70000]

plt.figure()
plt.title('Otomobil fiyat-KM Dağılım grafiği')
plt.xlabel('Km')
plt.ylabel('Fiyat')
plt.plot(x,y,'k.')
plt.axis([3000,20000,60000,95000])
plt.grid(True)
plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)

model_params = model.intercept_

coef = model.coef_

print(model_params)
print(coef)

plt.figure()
plt.title('Otomobilin Fiyat-KM Dağılım Grafiği')
plt.xlabel('KM')
plt.ylabel('Fiyat')
plt.plot(x,model.predict(x),color='red')
plt.plot(x,y,'k.')
plt.grid(True)
plt.show()


test_araba = np.array([[12000]])
predicted_price = model.predict(test_araba)[0]
print(f'1200 kmdeki aracın tahmini fiyatı: %.2f' % predicted_price)

y_predictions = model.predict(x)
for i,prediction in enumerate(y_predictions):
    print('Tahmin edilen fiyat: %.2f, Gerçek fiyatı: %s' % (prediction,y[i]))

