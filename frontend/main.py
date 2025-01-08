import streamlit as st
import requests

API_BASE_URL = "http://localhost:5000"

st.set_page_config(page_title="Shroomlense", layout="wide", initial_sidebar_state="collapsed")


def fetch_mushroom_image(mushroom_name: str):
    image_url = f"{API_BASE_URL}/internal/random_image/{mushroom_name}"
    image = requests.get(image_url).content

    return image


@st.fragment
def update_top_prediction_mushroom():
    top_mushroom_image = fetch_mushroom_image(top_mushroom_name)
    st.image(
        top_mushroom_image,
        caption=f"{top_mushroom_name.replace('_', ' ')} ({top_prediction['confidence']:.2%})",
        use_container_width=True,
    )

    st.button("🔄 Reload Image")


@st.dialog("Details")
def details(mushroom_name):
    st.markdown(f"<span style=font-size:25px>{mushroom_name}</span>", unsafe_allow_html=True)
    info_url = f"{API_BASE_URL}/external/wikipedia/{mushroom_name}/summary"
    prediction_text = requests.get(info_url).json()
    st.markdown(prediction_text)


@st.fragment
def show_details(name):
    if st.button("See more", key=f"{name}"):
        details(name)


st.markdown(
    """
    <style>
    .stImage { margin: auto; }
    .block-text { text-align: justify; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("\U0001F344 Shroomlense")
st.markdown("### Upload an image to classify and explore the results!")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file:
    classify_url = f"{API_BASE_URL}/internal/predict"
    files = {"image": uploaded_file}
    response = requests.post(classify_url, files=files).json()

    sorted_classes = sorted(response["predictions"], key=lambda x: x["confidence"], reverse=True)
    top_prediction = sorted_classes[0]
    other_predictions = sorted_classes[0:]

    top_mushroom_name = top_prediction["class_name"]

    text_url = f"{API_BASE_URL}/external/wikipedia/{top_mushroom_name}/summary"
    top_text = requests.get(text_url).json()

    left, center, right = st.columns([1, 2, 1])

    with left:
        st.markdown("### 🎯 Top Match")
        update_top_prediction_mushroom()

    with center:
        st.markdown(f"<strong style='padding-left: 5rem; font-size: 28px;'>📝 Details</strong>", unsafe_allow_html=True)
        info_name = top_mushroom_name.replace("_", " ")
        st.markdown(f"<span style=padding-left:5rem;font-size:25px>{info_name}</span>", unsafe_allow_html=True)
        st.markdown(
            f"<span style=padding-left:5rem;font-size:20px>Confidence: {top_prediction['confidence']:.2%}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div style='padding-left:5rem;padding-right:5rem'>{top_text}</div>", unsafe_allow_html=True)

    with right:
        st.markdown("### \U0001F50E Other Matches")
        for prediction in other_predictions[1:5]:
            mushroom_image = fetch_mushroom_image(prediction["class_name"])
            st.image(
                mushroom_image,
                caption=f"{prediction['class_name'].replace('_', ' ')} ({prediction['confidence']:.2%})",
                width=220,
            )
            show_details(prediction["class_name"])
