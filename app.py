import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "final_saved_model"
CLASS_NAMES = ["apple", "banana", "orange"]
IMG_SIZE = (128, 128)

st.title("Fruit Freshness Classifier")

st.write(
    "Upload an image of an apple, banana, or orange, "
    "and the model will predict which fruit it is."
)

st.info(
    "Supported fruits: Apple, Banana, and Orange. "
    "For best results, use a clear image with the fruit visible."
)


@st.cache_resource
def load_model():
    model = tf.saved_model.load(MODEL_PATH)
    return model.signatures["serving_default"]


model = load_model()

st.subheader("1. Upload an image")

uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG image of an apple, banana, or orange."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True,
    )

    st.subheader("2. Run the prediction")

    if st.button("Classify fruit", type="primary"):
        with st.spinner("Analyzing image..."):
            image_array = np.array(image)

            image_tensor = tf.convert_to_tensor(image_array)
            image_tensor = tf.image.resize_with_pad(
                image_tensor,
                IMG_SIZE[0],
                IMG_SIZE[1],
            )
            image_tensor = tf.cast(image_tensor, tf.float32)
            image_tensor = tf.expand_dims(image_tensor, axis=0)

            predictions = model(image_tensor)
            probabilities = next(iter(predictions.values())).numpy()[0]

            predicted_index = np.argmax(probabilities)
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = probabilities[predicted_index]

        st.subheader("3. Prediction result")

        st.success(f"Predicted fruit: **{predicted_class.capitalize()}**")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}",
        )

        st.write("Prediction probabilities:")

        for class_name, probability in zip(
            CLASS_NAMES, probabilities
        ):
            st.write(f"**{class_name.capitalize()}**")
            st.progress(float(probability))
            st.caption(f"{probability:.2%}")

else:
    st.caption("Your image preview and prediction will appear here after uploading.")