
import streamlit as st
from pathlib import Path


SCORE_LABELS = {
    "keyword": "BM25 score",
    "fuzzy": "Fuzzy score",
    "semantic": "Cosine similarity",
    "visual": "Cosine similarity",
    "hybrid": "RRF score",
    "multimodal": "Cosine similarity"
}


def render_products(products, mode, columns=4):

    if not products:
        st.info("No matching products found.")
        return

    grid = st.columns(columns)

    for index, product in enumerate(products):

        with grid[index % columns]:

            with st.container(border=True):

                st.caption(f"Rank #{index + 1}")

                image_path = Path(product["image_path"])

                if image_path.is_file():
                    st.image(
                        str(image_path),
                        width="stretch"
                    )
                else:
                    st.caption("Image unavailable")

                st.text(product["name"])

                st.caption(
                    f"{product['article_type']} | "
                    f"{product['colour']}"
                )

                label = SCORE_LABELS.get(mode, "Score")

                st.caption(
                    f"{label}: {product['score']:.4f}"
                )

