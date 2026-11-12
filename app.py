
import json
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Kucing dan Anjing",
    page_icon="🐱",
    layout="centered"
)

# Judul website
st.title("🐱 Klasifikasi Kucing dan Anjing 🐶")
st.write(
    "Unggah gambar kucing atau anjing untuk mengetahui "
    "hasil prediksi menggunakan model Custom CNN."
)

# Memuat model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cat_dog_cnn.keras")

    with open("class_names.json", "r") as file:
        class_names = json.load(file)

    return model, class_names

try:
    model, class_names = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Mengubah nama kelas menjadi Bahasa Indonesia
def translate_label(label):
    label = label.lower()

    if label in ["cat", "cats"]:
        return "Kucing"
    elif label in ["dog", "dogs"]:
        return "Anjing"

    return label.capitalize()

# Upload gambar
uploaded_file = st.file_uploader(
    "Pilih gambar kucing atau anjing",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Membaca gambar
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diunggah",
        use_container_width=True
    )

    if st.button("🔍 Prediksi Gambar", use_container_width=True):

        with st.spinner("Sedang menganalisis gambar..."):

            # Menyesuaikan ukuran gambar dengan model
            input_shape = model.input_shape
            height = input_shape[1]
            width = input_shape[2]

            resized_image = image.resize((width, height))

            # Mengubah gambar menjadi array
            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            image_array = np.expand_dims(image_array, axis=0)

            # Prediksi
            prediction = model.predict(image_array, verbose=0)[0]

            predicted_index = int(np.argmax(prediction))
            predicted_class = class_names[predicted_index]
            confidence = float(prediction[predicted_index]) * 100

        # Menampilkan hasil
        result = translate_label(predicted_class)

        st.success("Prediksi berhasil!")

        st.subheader(f"Hasil Prediksi: {result}")
        st.metric(
            label="Tingkat Keyakinan",
            value=f"{confidence:.2f}%"
        )

        st.write("### Detail Probabilitas")

        for index, probability in enumerate(prediction):
            label = translate_label(class_names[index])
            percentage = float(probability) * 100

            st.write(f"**{label}: {percentage:.2f}%**")
            st.progress(float(probability))

st.divider()

st.caption(
    "Proyek Klasifikasi Gambar Kucing dan Anjing "
    "Menggunakan Custom CNN."
)
