import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Example true and predicted labels
y_true = [0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 0, 0, 0, 1]

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Draw heatmap
plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,        # show numbers
    fmt='d',           # integer format
    cmap='Blues',
    xticklabels=['Pred 0', 'Pred 1'],
    yticklabels=['True 0', 'True 1']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
