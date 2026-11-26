from functools import partial

import streamlit as st

from app.search.engine import SearchEngine
from ui import discovery, engineering_lab
from ui.components import apply_styles


st.set_page_config(page_title="FashionLens", layout="wide")


@st.cache_resource
def load_engine():
    return SearchEngine()


apply_styles()

with st.container(key="site_header"):
    st.html('<div class="fl-wordmark">FASHION<span>LENS</span></div>')

with st.spinner("Loading FashionLens…"):
    engine = load_engine()

page = st.navigation(
    [
        st.Page(partial(discovery.render, engine), title="FashionLens", default=True),
        st.Page(
            partial(engineering_lab.render, engine),
            title="Engineering Lab",
            url_path="engineering-lab",
        ),
    ],
    position="hidden",
)
page.run()
