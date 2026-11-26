from functools import partial

import streamlit as st

from app.search.engine import SearchEngine
from ui import discovery, engineering_lab
from ui.components import apply_styles


st.set_page_config(page_title="Fashion Lens", layout="wide")


@st.cache_resource
def load_engine():
    return SearchEngine()


apply_styles()

with st.spinner("Loading Fashion Lens…"):
    engine = load_engine()

home = st.Page(partial(discovery.render, engine), title="Fashion Lens", default=True)
page = st.navigation(
    [
        home,
        st.Page(
            partial(engineering_lab.render, engine),
            title="Engineering Lab",
            url_path="engineering-lab",
        ),
    ],
    position="hidden",
)
with st.container(key="site_header"):
    st.page_link(home, label="**Fashion** Lens")

page.run()
