import logging
from html import escape
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from app.search.catalog import FILTER_FIELDS, NAVIGATION
from ui.components import render_products


MODES = ["Keyword", "Fuzzy", "Semantic", "Visual", "Hybrid", "Multimodal"]
IMAGE_MODES = {"Visual", "Multimodal"}


def prepare_image(upload):
    if upload.size > 5 * 1024 * 1024:
        raise ValueError("Maximum image size is 5 MB.")

    try:
        with Image.open(BytesIO(upload.getvalue())) as source:
            if source.width * source.height > 20_000_000:
                raise ValueError("Image resolution is too large.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1024, 1024))
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            return buffer.getvalue()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise ValueError("Choose a valid JPG or PNG image.") from error


def run_search(engine, request, collection, filters, limit):
    with TemporaryDirectory() as directory:
        image_path = None
        if request["image"] is not None:
            image_path = Path(directory) / "query.jpg"
            image_path.write_bytes(request["image"])
        return engine.search(
            mode=request["mode"].lower(),
            text=request["text"],
            image_path=image_path,
            image_weight=request["weight"],
            collection=collection,
            filters=filters,
            limit=limit,
        )


def open_visual_search():
    st.session_state.discovery_mode = "Visual"


def reset_search():
    for key in ("discovery_request", "discovery_result"):
        st.session_state.pop(key, None)
    st.session_state.discovery_form_version = st.session_state.get("discovery_form_version", 0) + 1


def render(engine):
    current_request = st.session_state.get("discovery_request", {})
    form_version = st.session_state.get("discovery_form_version", 0)
    with st.container(key="search_area"):
        search, visual = st.columns(2, vertical_alignment="top")
        with visual:
            st.button(
                "Search by image", on_click=open_visual_search,
                icon=":material/photo_camera:", width="stretch",
            )

        with search:
            if "discovery_mode" not in st.session_state:
                st.session_state.discovery_mode = current_request.get("mode", "Hybrid")
            mode = st.session_state.discovery_mode
            with st.form("discovery_search", border=False):
                field, action = st.columns(2, vertical_alignment="top")
                with field:
                    text = ""
                    uploaded = None
                    if mode != "Visual":
                        text = st.text_input(
                            "Search products", value=current_request.get("text", ""),
                            placeholder="Search the catalog, e.g. black leather handbag",
                            label_visibility="collapsed", max_chars=500, key=f"discovery_query_{form_version}",
                        )
                    if mode in IMAGE_MODES:
                        uploaded = st.file_uploader(
                            "Product image", type=["jpg", "jpeg", "png"], key=f"discovery_upload_{form_version}",
                        )
                with action:
                    submitted = st.form_submit_button("Search", type="primary")

    with st.container(key="collection_nav"):
        collection = st.segmented_control(
            "Browse collections", list(NAVIGATION), default="Discover", required=True,
            label_visibility="collapsed", key="discovery_collection", persist_state="session",
        )

    options = engine.catalog.filter_options()
    with st.container(key="catalog_filters"):
        *filter_columns, settings = st.columns(5)
        filters = {}
        for column, (field, label) in zip(filter_columns, FILTER_FIELDS.items()):
            with column:
                filters[field] = st.selectbox(
                    label, options[field], index=None, placeholder="Any",
                    key=f"discovery_filter_{field}", persist_state="session",
                )
        with settings:
            with st.popover("Search settings", icon=":material/tune:", width="stretch"):
                st.selectbox("Search mode", MODES, key="discovery_mode", persist_state="session")
                limit = st.slider(
                    "Results to show", 4, 24, 12, step=4, key="discovery_limit",
                    persist_state="session",
                )
                weight = 0.5
                if mode == "Multimodal":
                    weight = st.slider(
                        "Image weight", 0.0, 1.0, 0.5, 0.1,
                        key="discovery_weight", persist_state="session",
                    )
                    st.caption(f"Text weight · {1 - weight:.0%}")

    if submitted:
        try:
            if mode != "Visual" and not text.strip():
                raise ValueError("Enter a search query.")
            if mode in IMAGE_MODES and uploaded is None:
                raise ValueError("Choose a product image.")
            st.session_state.discovery_request = {
                "mode": mode,
                "text": text.strip(),
                "image": prepare_image(uploaded) if uploaded is not None else None,
                "filename": uploaded.name if uploaded is not None else "",
                "weight": weight,
            }
            st.session_state.pop("discovery_result", None)
        except ValueError as error:
            st.warning(str(error))

    request = st.session_state.get("discovery_request")
    if request:
        signature = (collection, tuple(filters.items()), limit)
        result = st.session_state.get("discovery_result")
        if result is None or result[0] != signature:
            try:
                with st.spinner("Searching the catalog…"):
                    products = run_search(engine, request, collection, filters, limit)
                st.session_state.discovery_result = (signature, products)
            except ValueError as error:
                st.warning(str(error))
                return
            except Exception:
                st.error("The search could not be completed. Try again.")
                logging.exception("Discovery search failed")
                return
        else:
            products = result[1]

        query = request["text"]
        filename = request["filename"]
        title = f'Results for “{query}”' if query else f"Results for {filename}"
        if query and filename:
            title += f" + {filename}"
    else:
        products = engine.catalog.matching_products(collection, filters, limit)
        title = "Explore the catalog" if collection == "Discover" else collection

    count = f'{len(products)} {"result" if len(products) == 1 else "results"} shown'
    with st.container(key="results_header", horizontal=True,
                      horizontal_alignment="distribute", vertical_alignment="center"):
        st.html(
            f'<div class="fl-results-heading"><h1>{escape(title)}</h1>'
            f'<span>{count}</span></div>'
        )
        if request:
            st.button("Clear search", on_click=reset_search)

    render_products(products)
