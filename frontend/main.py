import streamlit as st
import requests

API_BASE_URL = "http://localhost:5000"

st.set_page_config(page_title="Shroomlense", layout="wide", initial_sidebar_state="collapsed")

def fetch_mushroom_image(mushroom_name: str) -> bytes:
    image_url = f"{API_BASE_URL}/internal/random_image/{mushroom_name}"
    image = requests.get(image_url).content

    return image

@st.fragment
def reload_image(mushroom_name: str) -> None:
    mushroom_img = fetch_mushroom_image(mushroom_name)
    st.image(
        mushroom_img,
        caption=f"{mushroom_name.replace('_', ' ')} ({top_prediction['confidence']:.2%})",
        use_container_width=True,
    )
    st.button("🔄 Reload Image")
    
@st.dialog("Summary")
def details(mushroom_name: str, image: bytes) -> None:
    left_side, right_side = st.columns([1, 1])

    with left_side:
        st.image(image)

    with right_side:
        st.html(f"<span style='font-size: 25px;'>{mushroom_name.replace('_', ' ')}</span>")

    info_url = f"{API_BASE_URL}/external/wikipedia/{mushroom_name}/summary"
    prediction_text = requests.get(info_url).json()
    st.html(f"<div style='text-align: justify;'>{prediction_text}</div>")


@st.dialog("Taxonomy")
def taxonomy(mushroom_name: str) -> None:
    st.html(f"<span style='font-size: 25px;'>{mushroom_name.replace('_', ' ')}</span>")
    taxonomy_url = f"{API_BASE_URL}/external/wikipedia/{mushroom_name}/table"
    taxonomy_text = requests.get(taxonomy_url).json()
    table_data = [[key, value] for key, value in taxonomy_text.items()]

    html_table = "<table>"
    for row in table_data:
        html_table += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    html_table += "</table>"
    st.markdown(html_table, unsafe_allow_html=True)


@st.fragment
def show_details(mushroom_name: str, image: bytes) -> None:
    left_col, right_col = st.columns(2)

    with left_col:
        if st.button("Summary", key=f"{mushroom_name}"):
            details(mushroom_name, image)
    with right_col:
        st.html("<span id='button-right'></span>")
        if st.button("Taxonomy", key=f"taxonomy_{mushroom_name}"):
            taxonomy(mushroom_name)


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
        .element-container:has(#button-right) + div button { float: right; }
        .headline { font-size: 28px; }
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

    left, center, right = st.columns([3, 4, 2], gap="large")

    with left:
        st.html(f"<strong style='font-size: 28px;'>🎯 Top Match</strong>")
        st.markdown('<div class="main-image">', unsafe_allow_html=True)
        reload_image(top_mushroom_name)
        st.markdown('</div>', unsafe_allow_html=True)

    with center:
        st.html(f"<strong style='font-size: 28px;'>📝 Details</strong>")

        info_name = top_mushroom_name.replace("_", " ")
        st.html(f"<span class='headline'>{info_name}</span>")
        st.html(f"<span style='font-size: 20px'>Confidence: {top_prediction['confidence']:.2%}</span>")

        if response.get("warning"):
            st.html(
                f"<span style='font-size: 16px; text-align: justify; color: #ffc107;'>"
                + "\U000026A0 WARNING! " + response["warning"] + "</span>"
            )

        st.html(f"<div style='text-align: justify;'>{top_text}</div>")

    with right:
        st.html(f"<strong style='font-size: 28px;'>\U0001F50E Other Matches</strong>")
        image_list = []
        for prediction in other_predictions[0:4]:
            mushroom_image = fetch_mushroom_image(prediction["class_name"])
            st.image(
                mushroom_image,
                caption=f"{prediction['class_name'].replace('_', ' ')} ({prediction['confidence']:.2%})",
                use_container_width=True,
            )
            show_details(prediction["class_name"], mushroom_image)