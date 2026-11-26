
import time
import tempfile

from pathlib import Path
from io import BytesIO
from PIL import Image

import streamlit as st

from ui.components import render_products


MODES = {
    "Keyword": "keyword",
    "Fuzzy": "fuzzy",
    "Semantic": "semantic",
    "Visual": "visual",
    "Hybrid": "hybrid",
    "Multimodal": "multimodal"
}

DESCRIPTIONS = {
    "keyword": "Search using product names and attributes.",
    "fuzzy": "Search with spelling mistake tolerance.",
    "semantic": "Find product images using natural language.",
    "visual": "Find products similar to an uploaded image.",
    "hybrid": "Combine keyword and semantic search.",
    "multimodal": "Combine an image and a text description."
}


def render(engine):

    st.header("Discover your next favourite")

    st.caption(
        "Explore fashion through text, images and AI."
    )

    selected = st.segmented_control(
        "Search mode",
        options=list(MODES.keys()),
        default="Hybrid"
    )

    mode = MODES[selected or "Hybrid"]

    st.info(DESCRIPTIONS[mode])

    with st.sidebar:

        st.subheader("Discovery Settings")

        limit = st.slider(
            "Number of results",
            min_value=4,
            max_value=24,
            value=12,
            step=4
        )

        image_weight = 0.5

        if mode == "multimodal":

            image_weight = st.slider(
                "Image importance",
                0.0, 1.0, 0.5, 0.1
            )

            st.caption(
                f"Text importance: "
                f"{1 - image_weight:.0%}"
            )

    text = None
    uploaded = None

    with st.form("discovery_search"):

        if mode != "visual":

            text = st.text_input(
                "Search products",
                placeholder="e.g. Black leather handbag",
                max_chars=500
            )

        if mode in ("visual", "multimodal"):

            uploaded = st.file_uploader(
                "Upload a product image",
                type=["jpg", "jpeg", "png"]
            )

        submitted = st.form_submit_button(
            "Search products",
            type="primary",
            width="stretch"
        )

    if submitted:

        temp_path = None

        st.session_state.pop("discovery_results", None)

        try:

            if uploaded is not None:

                if uploaded.size > 5 * 1024 * 1024:
                    raise ValueError(
                        "Maximum image size is 5 MB."
                    )

                with Image.open(
                    BytesIO(uploaded.getvalue())
                ) as source:

                    if source.width * source.height > 20000000:
                        raise ValueError(
                            "Image resolution is too large."
                        )

                    image = source.convert("RGB")
                    image.thumbnail((1024, 1024))

                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False
                ) as temp:

                    temp_path = temp.name
                    image.save(temp, format="JPEG")

            with st.spinner("Searching the catalog..."):

                start = time.perf_counter()

                results = engine.search(
                    mode=mode,
                    text=text,
                    image_path=temp_path,
                    image_weight=image_weight,
                    limit=limit
                )

                latency = (
                    time.perf_counter() - start
                ) * 1000

            st.session_state["discovery_results"] = {
                "products": results,
                "mode": mode,
                "label": selected or "Hybrid",
                "latency": latency
            }

        except ValueError as error:
            st.warning(str(error))

        except Exception:
            st.error("Search failed. Check the application logs.")
            import logging
            logging.exception("Discovery search failed")

        finally:

            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    result = st.session_state.get("discovery_results")

    if result:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Results",
            len(result["products"])
        )

        c2.metric(
            "Search latency",
            f"{result['latency']:.1f} ms"
        )

        c3.metric(
            "Search mode",
            result["label"]
        )

        st.divider()

        render_products(
            result["products"],
            result["mode"]
        )

