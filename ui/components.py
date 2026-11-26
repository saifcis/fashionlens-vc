from base64 import b64encode
from html import escape
from mimetypes import guess_type
from pathlib import Path

import streamlit as st


SCORE_LABELS = {
    "keyword": "BM25 score",
    "fuzzy": "Fuzzy score",
    "semantic": "Cosine similarity",
    "visual": "Cosine similarity",
    "hybrid": "RRF score",
    "multimodal": "Cosine similarity",
}


def apply_styles():
    st.html(Path(__file__).with_name("styles.css"))


@st.cache_data(max_entries=1024, show_spinner=False)
def image_source(path, modified_at):
    media_type = guess_type(path.name)[0] or "image/jpeg"
    content = b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{content}"


def product_card(product, path, score_mode):
    name = escape(product["name"])
    details = " · ".join(
        escape(value)
        for value in (product["article_type"], product["colour"])
        if value
    )
    if path.is_file():
        source = image_source(path, path.stat().st_mtime_ns)
        photo = f'<img src="{source}" alt="{name}" loading="lazy">'
    else:
        photo = "<span>Image unavailable</span>"

    score = ""
    if score_mode:
        label = SCORE_LABELS[score_mode]
        score = f'<p class="fl-product-meta">{label} · {product["score"]:.4f}</p>'

    return f"""
        <article class="fl-product">
            <div class="fl-product-image">{photo}</div>
            <h3>{name}</h3>
            <p class="fl-product-meta">{details}</p>
            {score}
        </article>
    """


def render_products(products, *, score_mode=None, on_find_similar=None):
    if not products:
        st.html('<p class="fl-empty">No matching products found.</p>')
        return

    if on_find_similar is None:
        cards = [
            product_card(product, Path(product["image_path"]), score_mode)
            for product in products
        ]
        st.html(f'<div class="fl-grid fl-grid-lab">{"".join(cards)}</div>')
        return

    with st.container(key="product_grid"):
        for product in products:
            path = Path(product["image_path"])
            with st.container():
                st.html(product_card(product, path, score_mode))
                st.button(
                    "Find similar",
                    key=f"find_similar_{product['id']}",
                    on_click=on_find_similar,
                    args=(product,),
                    disabled=not path.is_file(),
                )
