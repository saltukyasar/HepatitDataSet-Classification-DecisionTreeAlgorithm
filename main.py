"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Veri kümesini yükle
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
names = ['Class', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 'LiverBig', 'LiverFirm', 'SpleenPalpable', 'Spiders', 'Ascites', 'Varices', 'Bilirubin', 'AlkPhosphate', 'SGOT', 'Albumin', 'Protime', 'Histology']
data = pd.read_csv(url, names=names, na_values='?')


# Eksik verileri sil
data.dropna(inplace=True)

# Hedef değişkeni ayır
X = data.drop('Class', axis=1)
y = data['Class']

# Eğitim ve test verilerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree modelini eğit
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Test verileriyle sınıflandırma yap
y_pred = clf.predict(X_test)

# Performans ölçümleri
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy: {:.4f}".format(acc))
print("Precision: {:.4f}".format(prec))
print("Recall: {:.4f}".format(rec))
print("F1 Score: {:.4f}".format(f1))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

sns.histplot(data, x='Age', hue='Class', kde=True)
plt.title('Distribution of Age')
plt.show()
"""
import numpy as np

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# Veri kümesini yükle
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
names = ['Class', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 'LiverBig', 'LiverFirm', 'SpleenPalpable', 'Spiders', 'Ascites', 'Varices', 'Bilirubin', 'AlkPhosphate', 'SGOT', 'Albumin', 'Protime', 'Histology']
data = pd.read_csv(url, names=names, na_values='?')

# Eksik verileri sil
data.dropna(inplace=True)

# Hedef değişkeni ayır
X = data.drop('Class', axis=1)
y = data['Class']

# Eğitim ve test verilerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree modelini eğit
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Test verileriyle sınıflandırma yap
y_pred = clf.predict(X_test)

# Performans ölçümleri
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy: {:.4f}".format(acc))
print("Precision: {:.4f}".format(prec))
print("Recall: {:.4f}".format(rec))
print("F1 Score: {:.4f}".format(f1))
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

# Veri kümesini yükle
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
names = ['Class', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 'LiverBig', 'LiverFirm', 'SpleenPalpable', 'Spiders', 'Ascites', 'Varices', 'Bilirubin', 'AlkPhosphate', 'SGOT', 'Albumin', 'Protime', 'Histology']
data = pd.read_csv(url, names=names, na_values='?')

# Eksik verileri sil
data.dropna(inplace=True)

# Hedef değişkeni ayır
X = data.drop('Class', axis=1)
y = data['Class']

# Eğitim ve test verilerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree modelini eğit
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Test verileriyle sınıflandırma yap
y_pred = clf.predict(X_test)

# Karar ağacı modelinin oluşturulması
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Test verileri ile modelin performansının değerlendirilmesi
accuracy = decision_tree.score(X_test, y_test)
print("Accuracy:", accuracy)



# Confusion Matrix'i hesapla
cm = confusion_matrix(y_test, y_pred)

# True Positive Rate, True Negative Rate,
# False Positive Rate, False Negative Rate hesapla
tn, fp, fn, tp = cm.ravel()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

# Performans ölçümleri
acc = accuracy_score(y_test, y_pred)
prec = tp / (tp + fp)
rec = tp / (tp + fn)
f1 = (2*prec*rec)/(prec+rec)

print("Accuracy: {:.4f}".format(acc))
print("Precision: {:.4f}".format(prec))
print("Recall: {:.4f}".format(rec))
print("F1 Score: {:.4f}".format(f1))
print("True Positive Rate: {:.4f}".format(tpr))
print("True Negative Rate: {:.4f}".format(tnr))
print("False Positive Rate: {:.4f}".format(fpr))
print("False Negative Rate: {:.4f}".format(fnr))

# y_test sınıf etiketlerini 0 ve 1 olarak kodla
y_test = y_test.map({1: 0, 2: 1})

# Test verileri için olasılık tahminlerini hesapla
y_prob = clf.predict_proba(X_test)[:, 1]

# ROC eğrisini çiz
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# ROC eğrisini çizdir
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Histogram
plt.figure(figsize=(10,5))
sns.histplot(data=data, x='Age', kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Two-Dimensional Histogram
plt.figure(figsize=(10,5))
sns.histplot(data=data, x='Age', y='Bilirubin', bins=20)
plt.title('Age vs Bilirubin')
plt.xlabel('Age')
plt.ylabel('Bilirubin')
plt.show()

# Box Plot
plt.figure(figsize=(10,5))
sns.boxplot(data=data[['Bilirubin', 'AlkPhosphate', 'SGOT']])
plt.title('Boxplot of Bilirubin, AlkPhosphate, and SGOT')
plt.show()

# Scatter Plot
plt.figure(figsize=(10,5))
sns.scatterplot(data=data, x='Bilirubin', y='SGOT', hue='Class')
plt.title('Bilirubin vs SGOT')
plt.xlabel('Bilirubin')
plt.ylabel('SGOT')
plt.show()

# Matrix Plot
plt.figure(figsize=(10,5))
sns.pairplot(data=data, hue='Class')
plt.show()

# Correlation Matrix
plt.figure(figsize=(10,5))
sns.heatmap(data.corr(), annot=True, cmap='Blues')
plt.title('Correlation Matrix')
plt.show()

# Parallel Coordinates Plot
from pandas.plotting import parallel_coordinates
plt.figure(figsize=(10,5))
parallel_coordinates(data, 'Class')
plt.show()

# Star Plot
from pandas.plotting import radviz
plt.figure(figsize=(10,5))
radviz(data, 'Class')
plt.show()

# Tahmin olasılıklarını hesapla
y_prob = clf.predict_proba(X_test)[:,1]
y_true = np.where(y_test == 1, 0, 1)

# ROC eğrisini hesapla
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# AUC hesapla
auc = roc_auc_score(y_test, y_prob)

# ROC eğrisini görselleştir
plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# Korelasyon matrisi hesapla
corr_matrix = data.corr()

# Heatmap görselleştirme
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap='Blues')
plt.title('Correlation Matrix')
plt.show()

