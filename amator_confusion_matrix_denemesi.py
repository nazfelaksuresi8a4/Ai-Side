import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import confusion_matrix

ndarr = confusion_matrix(y_true=np.random.normal(0,10,100).astype(np.int16),y_pred=np.random.normal(0,10,100).astype(np.int16))

sns.heatmap(data=ndarr,
            vmin=0.5,
            vmax=1.0,
            annot=True)

plt.plot(ndarr)
plt.show()
