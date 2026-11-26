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
            exclude_product_id=request.get("product_id"),
        )


def open_visual_search():
    st.session_state.discovery_mode = "Visual"


def reset_search():
    for key in ("discovery_request", "discovery_result", "discovery_image", "discovery_image_error"):
        st.session_state.pop(key, None)
    st.session_state.discovery_replacing = False
    st.session_state.discovery_form_version = st.session_state.get("discovery_form_version", 0) + 1


def replace_image(mode):
    st.session_state.discovery_mode = mode
    st.session_state.discovery_replacing = True
    st.session_state.discovery_upload_version = st.session_state.get("discovery_upload_version", 0) + 1


def cancel_replacement():
    st.session_state.discovery_replacing = False
    st.session_state.pop("discovery_image_error", None)


def accept_upload(key):
    uploaded = st.session_state[key]
    if uploaded is None:
        return

    try:
        image = prepare_image(uploaded)
    except ValueError as error:
        st.session_state.discovery_image_error = str(error)
        return

    st.session_state.discovery_image = {"image": image}
    st.session_state.pop("discovery_image_error", None)
    st.session_state.discovery_replacing = False


def find_similar(product):
    try:
        image = Path(product["image_path"]).read_bytes()
    except OSError:
        st.session_state.discovery_image_error = "This product image is unavailable."
        return

    reset_search()
    open_visual_search()
    query_image = {
        "image": image,
        "product_id": product["id"],
    }
    st.session_state.discovery_image = query_image
    st.session_state.discovery_request = {
        **query_image, "mode": "Visual", "text": "", "weight": 0.5,
    }


def render_image_controls(mode, form_version):
    image = st.session_state.get("discovery_image")
    if image:
        with st.container(horizontal=True, vertical_alignment="center", key="query_image"):
            st.image(image["image"], width=64)
            st.markdown("**Query image**", width="content")
            if st.session_state.get("discovery_replacing"):
                st.button("Cancel", on_click=cancel_replacement)
            else:
                st.button("Replace", on_click=replace_image, args=(mode,))

    if not image or st.session_state.get("discovery_replacing"):
        upload_version = st.session_state.get("discovery_upload_version", 0)
        key = f"discovery_upload_{form_version}_{upload_version}"
        st.file_uploader(
            "Product image", type=["jpg", "jpeg", "png"], key=key,
            on_change=accept_upload, args=(key,),
        )


def render(engine):
    current_request = st.session_state.get("discovery_request", {})
    form_version = st.session_state.get("discovery_form_version", 0)
    if "discovery_mode" not in st.session_state:
        st.session_state.discovery_mode = current_request.get("mode", "Hybrid")
    mode = st.session_state.discovery_mode
    image = st.session_state.get("discovery_image") if mode in IMAGE_MODES else None
    needs_image = mode in IMAGE_MODES and (
        not image or st.session_state.get("discovery_replacing", False)
    )

    with st.container(key="search_area"):
        if mode in IMAGE_MODES:
            render_image_controls(mode, form_version)

        if error := st.session_state.get("discovery_image_error"):
            st.warning(error)

        search, visual = st.columns(2, vertical_alignment="top")
        with visual:
            if mode != "Visual":
                st.button(
                    "Search by image", on_click=open_visual_search,
                    icon=":material/photo_camera:", width="stretch",
                )

        with search, st.form("discovery_search", border=False):
            text = ""
            if mode == "Visual":
                submitted = st.form_submit_button("Search", type="primary", disabled=needs_image)
            else:
                field, action = st.columns(2, vertical_alignment="top")
                with field:
                    text = st.text_input(
                        "Search products", value=current_request.get("text", ""),
                        placeholder="Search the catalog, e.g. black leather handbag",
                        label_visibility="collapsed", max_chars=500, key=f"discovery_query_{form_version}",
                    )
                with action:
                    submitted = st.form_submit_button("Search", type="primary", disabled=needs_image)

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
        if mode != "Visual" and not text.strip():
            st.warning("Enter a search query.")
        elif needs_image:
            st.warning("Choose a product image.")
        else:
            st.session_state.discovery_request = {
                **(image or {"image": None}),
                "mode": mode,
                "text": text.strip(),
                "weight": weight,
            }
            st.session_state.pop("discovery_result", None)

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

    else:
        products = engine.catalog.matching_products(collection, filters, limit)

    render_results_header(request, collection, len(products))
    render_products(products, on_find_similar=find_similar)


def render_results_header(request, collection, count):
    image_search = request and request["image"] is not None
    mode = request["mode"] if request else None
    if mode == "Visual":
        title = "Visually similar products"
    elif request:
        title = f'Results for “{request["text"]}”'
    else:
        title = "Explore the catalog" if collection == "Discover" else collection

    noun = "product" if count == 1 else "products"
    summary = f"{count} matching {noun}" if image_search else (
        f'{count} {"result" if count == 1 else "results"} shown'
    )

    with st.container(key="results_header", horizontal=True,
                      horizontal_alignment="distribute", vertical_alignment="center"):
        with st.container(horizontal=True, vertical_alignment="center", width="content"):
            if image_search:
                st.image(request["image"], width=64)
            mode_label = f'<span>Search mode: {mode} · CLIP</span>' if image_search else ""
            st.html(
                f'<div class="fl-results-heading"><h1>{escape(title)}</h1>'
                f'<span>{summary}</span>{mode_label}</div>',
                width="content",
            )
        if request:
            with st.container(horizontal=True, vertical_alignment="center", width="content"):
                if image_search:
                    st.button("Change image", on_click=replace_image, args=(mode,))
                st.button("Clear" if image_search else "Clear search", on_click=reset_search)
