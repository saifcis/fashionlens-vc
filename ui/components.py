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


def render_products(products, *, score_mode=None):
    if not products:
        st.html('<p class="fl-empty">No matching products found.</p>')
        return

    cards = []
    for product in products:
        name = escape(product["name"])
        details = " · ".join(
            escape(value)
            for value in (product["article_type"], product["colour"])
            if value
        )
        path = Path(product["image_path"])
        if path.is_file():
            source = image_source(path, path.stat().st_mtime_ns)
            photo = f'<img src="{source}" alt="{name}" loading="lazy">'
        else:
            photo = "<span>Image unavailable</span>"

        score = ""
        if score_mode:
            label = SCORE_LABELS[score_mode]
            score = f'<p class="fl-product-meta">{label} · {product["score"]:.4f}</p>'

        cards.append(f"""
            <article class="fl-product">
                <div class="fl-product-image">{photo}</div>
                <h3>{name}</h3>
                <p class="fl-product-meta">{details}</p>
                {score}
            </article>
        """)

    layout = "fl-grid fl-grid-lab" if score_mode else "fl-grid"
    st.html(f'<div class="{layout}">{"".join(cards)}</div>')
