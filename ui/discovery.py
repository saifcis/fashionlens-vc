import logging
from html import escape
from pathlib import Path
from time import perf_counter

import streamlit as st

from app.search.catalog import FILTER_FIELDS, NAVIGATION
from app.search.relevance import RelevanceBlocked
from ui.components import render_products
from ui.query_image import prepare_image, prepare_query


MODES = ["Keyword", "Fuzzy", "Semantic", "Visual", "Hybrid", "Multimodal"]
IMAGE_MODES = {"Visual", "Multimodal"}


def run_search(engine, request, collection, filters, limit):
    relevance = engine.relevance
    previous_configuration = request.get("image_query_configuration")
    if request["image"] is not None and previous_configuration != relevance.configuration:
        if previous_configuration is not None:
            request.pop("override_relevance", None)
        request["image_query"] = prepare_query(
            engine, request["image"], st.session_state, "discovery_image_embedding",
        )
        request["image_query_configuration"] = relevance.configuration

    fallback_id = request.pop("image_fallback_query_id", None)
    start = perf_counter() if fallback_id else None
    products = None
    try:
        products = engine.search(
            mode=request["mode"].lower(),
            text=request["text"],
            image_query=request.get("image_query"),
            override_relevance=request.get("override_relevance", False),
            image_weight=request["weight"],
            collection=collection,
            filters=filters,
            limit=limit,
            exclude_product_id=request.get("product_id"),
        )
        return products
    finally:
        if fallback_id:
            relevance.record_action(
                fallback_id, "multimodal", "remove_image", "text_only_fallback",
                search_status="failed" if products is None else "completed",
                search_latency_ms=(perf_counter() - start) * 1000,
                result_count=None if products is None else len(products),
            )


def continue_without_image():
    request = st.session_state.discovery_request
    request.update(mode="Semantic", image=None)
    request["image_fallback_query_id"] = request.pop("image_query").query_id
    request.pop("override_relevance", None)
    request.pop("image_query_configuration", None)
    request.pop("product_id", None)
    request.pop("relevance_decision", None)
    st.session_state.discovery_mode = "Semantic"
    st.session_state.pop("discovery_image", None)
    st.session_state.pop("discovery_image_embedding", None)
    st.session_state.pop("discovery_image_error", None)
    st.session_state.pop("discovery_result", None)
    st.session_state.discovery_replacing = False
    st.session_state.discovery_upload_version = st.session_state.get("discovery_upload_version", 0) + 1


def confirm_image():
    st.session_state.discovery_request["override_relevance"] = True
    st.session_state.pop("discovery_result", None)


def render_relevance_warning(request, decision):
    st.warning("This may not be a fashion item.")
    st.caption("For better results, upload a photograph of clothing, shoes, bags or fashion accessories.")

    with st.container(horizontal=True):
        st.button("Upload another image", on_click=replace_image, args=(request["mode"],))
        if decision == "confirm" and not st.session_state.get("discovery_replacing"):
            st.button("Search anyway", type="primary", on_click=confirm_image)
        if request["mode"] == "Multimodal":
            st.button("Remove image and search", on_click=continue_without_image)


def open_visual_search():
    st.session_state.discovery_mode = "Visual"


def reset_search():
    for key in ("discovery_request", "discovery_result", "discovery_image",
                "discovery_image_embedding", "discovery_image_error"):
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
    request = st.session_state.get("discovery_request")
    st.session_state.pop("discovery_image_embedding", None)
    if request and request["image"] is not None:
        st.session_state.pop("discovery_request", None)
        st.session_state.pop("discovery_result", None)
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
        signature = (collection, tuple(filters.items()), limit, engine.relevance.configuration)
        result = st.session_state.get("discovery_result")
        if result is None or result[0] != signature:
            try:
                with st.spinner("Searching the catalog…"):
                    products = run_search(engine, request, collection, filters, limit)
                request.pop("relevance_decision", None)
                st.session_state.discovery_result = (signature, products)
            except RelevanceBlocked as error:
                request["relevance_decision"] = error.decision
                st.session_state.discovery_result = (signature, None)
                products = None
            except ValueError as error:
                st.warning(str(error))
                return
            except Exception:
                st.error("The search could not be completed. Try again.")
                logging.error("Discovery search failed")
                return
        else:
            products = result[1]

    else:
        products = engine.catalog.matching_products(collection, filters, limit)

    if products is None:
        render_relevance_warning(request, request["relevance_decision"])
        return

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
