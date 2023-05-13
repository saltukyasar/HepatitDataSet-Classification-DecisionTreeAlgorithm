import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# Veri setini yükle
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data", header=None)

# Sütun adlarını ayarla
df.columns = ["Class", "Age", "Sex", "Steroid", "Antivirals", "Fatigue", "Malaise", "Anorexia", "LiverBig", "LiverFirm", "SpleenPalpable", "Spiders", "Ascites", "Varices", "Bilirubin", "AlkPhosphate", "Sgot", "Albumin", "Protime", "Histology"]

# Eksik verileri belirle
missing_values = ["?", "NA", "--", "-"]
df = df.replace(missing_values, np.nan)
null_columns = df.columns[df.isnull().any()]
print("Eksik verilerin olduğu sütunlar:", null_columns)

# Eksik verilerin %30'dan fazla olduğu sütunları çıkar
df = df.dropna(thresh=0.7*len(df), axis=1)
print("Eksik verilerin %30'dan fazla olduğu sütunlar çıkarıldı.")

# Eksik verileri ortalama ile doldur
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)

# Karar ağacı modelini eğit
X = df.drop(["Class"], axis=1)
y = df["Class"]
dt = DecisionTreeClassifier()
dt.fit(X, y)
print("Karar ağacı modeli eğitildi.")

import missingno as msno
import matplotlib.pyplot as plt

# Eksik verilerin nerede olduğunu gösteren matrisi oluştur
msno.matrix(df)
plt.show()

