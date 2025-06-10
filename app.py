import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Modeli yükle
model = joblib.load("kacma_risk_modeli.pkl")

st.title("Çocuk İzinsiz Ayrılma Riski Değerlendirme Sistemi")

# Kullanıcıdan veri al
yas = st.slider("Yaş", 13, 18, 15)
gecmis_kacis = st.selectbox("Geçmişte izinsiz ayrıldı mı?", ["Hayır", "Evet"])
psikolojik_destek = st.selectbox("Psikolojik destek alıyor mu?", ["Hayır", "Evet"])
kurum_tipi = st.selectbox("Kurum tipi", ["Çocuk Evi", "Sevgi Evi"])
okul_devam = st.selectbox("Okula devam ediyor mu?", ["Hayır", "Evet"])
ailesiyle_iletisim = st.selectbox("Aile ile düzenli iletişimi var mı?", ["Hayır", "Evet"])

# Model için veriyi hazırla
veri = np.array([
    yas,
    1 if gecmis_kacis == "Evet" else 0,
    1 if psikolojik_destek == "Evet" else 0,
    1 if kurum_tipi == "Sevgi Evi" else 0,
    1 if okul_devam == "Evet" else 0,
    1 if ailesiyle_iletisim == "Evet" else 0,
]).reshape(1, -1)

# Tahmin ve açıklama
if st.button("Riski Tahmin Et"):
    tahmin = model.predict(veri)[0]
    olasilik = model.predict_proba(veri)[0][1] * 100

    # Açıklama nedenlerini topla
    nedenler = []
    if gecmis_kacis == "Evet":
        nedenler.append("çocuk daha önce kurumdan izinsiz ayrılmış")
    if psikolojik_destek == "Hayır":
        nedenler.append("psikolojik destek almıyor")
    if okul_devam == "Hayır":
        nedenler.append("okula devam etmiyor")
    if ailesiyle_iletisim == "Hayır":
        nedenler.append("ailesiyle düzenli iletişimi yok")
    if yas < 15:
        nedenler.append("yaşı küçük")

    # Risk yüzdesi
    st.write(f"Tahmini risk oranı: %{olasilik:.1f}")

    # Riske göre uyarı rengi
    if olasilik >= 80:
        st.error("⚠️ YÜKSEK RİSK!")
    elif 40 <= olasilik < 80:
        st.warning("⚠️ ORTA RİSK!")
    else:
        st.success("✅ DÜŞÜK RİSK.")

    # Nedenleri listele
    if nedenler:
        st.markdown("**Bu tahmin şu nedenlere dayanıyor:**")
        for neden in nedenler:
            st.markdown(f"- {neden}")

# --- Özellik önem grafiği ---
st.markdown("---")
st.subheader("📊 Modeldeki Değişkenlerin Önemi")

try:
    image = Image.open("feature_importance.png")
    st.image(image, caption="Özellik önem düzeyleri", use_column_width=True)
except FileNotFoundError:
    st.warning("Özellik önem grafiği bulunamadı. Lütfen önce modeli eğit.")
