import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib
import os

print("Kod başladı...")

# Veriyi yükle
df = pd.read_excel("ham_cocuk_verisi.xlsx")

# Hedef değişken üret (daha dengeli koşullarla)
df["kacma_riski"] = (
    (df["gecmis_kacis"] == 1) |
    ((df["ailesiyle_iletisim"] == 0) & (df["psikolojik_destek"] == 0)) |
    ((df["yas"] < 15) & (df["okul_devam"] == 0))
).astype(int)

# Özellikler ve hedef
X = df[["yas", "gecmis_kacis", "psikolojik_destek", "kurum_tipi", "okul_devam", "ailesiyle_iletisim"]]
y = df["kacma_riski"]

# Veriyi eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modeli
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Modeli kaydet
joblib.dump(model, "kacma_risk_modeli.pkl")

# Doğruluk çıktısı
y_pred = model.predict(X_test)
print("Doğruluk:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Özellik önemlerini görselleştir
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Özellik önemlerini ve isimleri al
importances = model.feature_importances_
feature_names = X.columns

# Veriyi görselleştirme için hazırlayalım
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Özellik önemlerini ve isimleri al
importances = model.feature_importances_
feature_names = X.columns

# Görselleştirme için dataframe
importance_df = pd.DataFrame({
    "Özellik": feature_names,
    "Önem Düzeyi": importances
}).sort_values("Önem Düzeyi", ascending=True)

# Grafik stili
sns.set(style="whitegrid", font_scale=1.1)

# Grafik çizimi
plt.figure(figsize=(10, 6), dpi=200)
ax = sns.barplot(
    data=importance_df,
    x="Önem Düzeyi",
    y="Özellik",
    color="steelblue"  # palette yerine tek renk
)

# Etiketleri ekle (manuel olarak çubukların üstüne)
for i in range(len(importance_df)):
    plt.text(
        importance_df["Önem Düzeyi"].iloc[i] + 0.005,
        i,
        f'{importance_df["Önem Düzeyi"].iloc[i]:.2f}',
        va='center',
        fontsize=10
    )

plt.title("Modeldeki Özelliklerin Göreli Önemi", fontsize=14)
plt.xlabel("Önem Düzeyi", fontsize=12)
plt.ylabel("")
plt.tight_layout()

# Kayıt
output_path = "feature_importance.png"
plt.savefig(output_path, bbox_inches="tight")
plt.savefig("feature_importance.svg", format='svg', bbox_inches='tight')
plt.savefig("feature_importance.pdf", format='pdf', bbox_inches='tight')

if os.path.exists(output_path):
    print(f"✅ Grafik başarıyla kaydedildi: {output_path}")
else:
    print("❌ Grafik oluşturulamadı.")

print("Model başarıyla eğitildi ve kaydedildi.")
