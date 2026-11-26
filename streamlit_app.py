
import streamlit as st

from app.search.engine import SearchEngine
from ui import discovery, engineering_lab


st.set_page_config(
    page_title="FashionLens",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def load_engine():
    return SearchEngine()


st.sidebar.title("FashionLens")

st.sidebar.caption(
    "Intelligent Fashion Discovery"
)

page = st.sidebar.radio(
    "Navigation",
    ["Discovery", "Engineering Lab"]
)

st.sidebar.divider()

st.sidebar.caption(
    "Independent ML engineering portfolio project."
)


with st.spinner("Initializing FashionLens..."):

    engine = load_engine()


st.title("FashionLens")

st.caption(
    "Multimodal Fashion Search & Discovery Engine"
)


if page == "Discovery":

    discovery.render(engine)

else:

    engineering_lab.render(engine)

