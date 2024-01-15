import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import streamlit as st

# interact with FastAPI endpoint
backend = "https://predict-3aoiym5c7a-lz.a.run.app/classify"


def process(image, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})
    r = requests.post(server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)

    # Decode the JSON response and extract the 'class' value
    response_json = r.json()
    predicted_class = response_json.get('class', 'No class found')  # Default message if 'class' key is not found
    return predicted_class



# construct UI layout
st.title("MRI Scan tumor classification")

st.write("Classify MRI scans into 4 classes: glioma, meningioma, pituitary tumor, no tumor.")  # description and instructions

input_image = st.file_uploader("insert image")  # image upload widget

if st.button("Classify MRI scan"):

    col1, col2 = st.columns(2)

    if input_image:
        result = process(input_image, backend)
        col1.header("Scan")
        col1.image(input_image, use_column_width=True)
        col2.header("Result")
        col2.text(result)

    else:
        # handle case with no image
        st.write("Insert an image!")
