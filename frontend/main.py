import streamlit as st
import requests
from streamlit import dialog

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


@st.dialog("Summary")
def details(mushroom_name, image):
    left_side, right_side = st.columns([1, 1])

    with left_side:
        st.image(image)

    with right_side:
        st.markdown(f"<span style='font-size: 25px;'>{mushroom_name.replace('_', ' ')}</span>", unsafe_allow_html=True)

    info_url = f"{API_BASE_URL}/external/wikipedia/{mushroom_name}/summary"
    prediction_text = requests.get(info_url).json()
    st.markdown(f"<div style='text-align: justify;'>{prediction_text}</div>", unsafe_allow_html=True)


@st.dialog("Taxonomy")
def taxonomy(mushroom_name):
    st.markdown(f"<span style='font-size: 25px;'>{mushroom_name.replace('_', ' ')}</span>", unsafe_allow_html=True)
    taxonomy_url = f"{API_BASE_URL}/external/wikipedia/{mushroom_name}/table"
    taxonomy_text = requests.get(taxonomy_url).json()
    table_data = [[key, value] for key, value in taxonomy_text.items()]

    html_table = "<table>"
    for row in table_data:
        html_table += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    html_table += "</table>"

    st.markdown(html_table, unsafe_allow_html=True)


@st.fragment
def show_details(mushroom_name, image):
    left_col, right_col, space = st.columns([1, 1, 2])
    with left_col:
        if st.button("Summary", key=f"{mushroom_name}"):
            details(mushroom_name, image)
    with right_col:
        if st.button("Taxonomy", key=f"taxonomy_{mushroom_name}"):
            taxonomy(mushroom_name)


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
css = """
   <style>
        [data-testid='stFileUploader'] {
            width: max-content;
        }
        [data-testid='stFileUploader'] section {
            padding: 0;
            float: left;
        }
        [data-testid='stFileUploader'] section > input + div {
            display: none;
        }
        [data-testid='stFileUploader'] section + div {
            float: right;
            padding-top: 0;
        }

    </style>
    """
st.markdown(css, unsafe_allow_html=True)

if uploaded_file:
    classify_url = f"{API_BASE_URL}/internal/predict"
    files = {"image": uploaded_file}
    response = requests.post(classify_url, files=files).json()

    sorted_classes = sorted(response["predictions"], key=lambda x: x["confidence"], reverse=True)
    top_prediction = sorted_classes[0]
    other_predictions = sorted_classes[1:]

    top_mushroom_name = top_prediction["class_name"]

    text_url = f"{API_BASE_URL}/external/wikipedia/{top_mushroom_name}/summary"
    top_text = requests.get(text_url).json()

    left, center, right = st.columns([0.4, 0.5, 0.3], vertical_alignment="top")

    with left:
        st.markdown("### 🎯 Top Match")
        update_top_prediction_mushroom()

    with center:
        st.markdown(f"<strong style='padding-left: 5rem; font-size: 28px;'>📝 Details</strong>", unsafe_allow_html=True)
        info_name = top_mushroom_name.replace("_", " ")
        st.markdown(f"<span style='padding-left: 5rem; font-size: 25px'>{info_name}</span>", unsafe_allow_html=True)
        st.markdown(
            f"<span style='padding-left: 5rem; font-size: 20px'>Confidence: {top_prediction['confidence']:.2%}</span>",
            unsafe_allow_html=True,
        )
        if top_prediction["confidence"] < 0.6 or sum([pred["confidence"] for pred in sorted_classes]) < 0.9:
            st.markdown(
                f"<span style='margin-left: 5rem; font-size: 18px; background-color: #ffc107;'>\U000026A0 WARNING! Low confidence in top 5 predictions</span>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div style='padding-left: 5rem; padding-right: 5rem; text-align: justify;'>{top_text}</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### \U0001F50E Other Matches")
        image_list = []
        for prediction in other_predictions[0:4]:
            mushroom_image = fetch_mushroom_image(prediction["class_name"])
            st.image(
                mushroom_image,
                caption=f"{prediction['class_name'].replace('_', ' ')} ({prediction['confidence']:.2%})",
                width=220,
            )
            show_details(prediction["class_name"], mushroom_image)
