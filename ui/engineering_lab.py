
import time

import pandas as pd
import streamlit as st

from ui.components import render_products


SEARCH_MODES = {
    "Keyword (BM25)": "keyword",
    "Fuzzy (RapidFuzz)": "fuzzy",
    "Semantic (CLIP)": "semantic",
    "Hybrid (RRF)": "hybrid"
}


def render(engine):

    st.html('<div class="fl-results-heading"><h1>Engineering Lab</h1></div>')

    st.caption(
        "Compare search mechanisms, performance "
        "and retrieved products."
    )

    tab1, tab2 = st.tabs([
        "Search Comparison",
        "System Overview"
    ])

    with tab1:

        with st.form("comparison_form"):

            query_column, limit_column = st.columns([3, 1])
            with query_column:
                query = st.text_input(
                    "Comparison query",
                    value="black leather handbag",
                    max_chars=500
                )
            with limit_column:
                top_k = st.slider("Results per mechanism", 3, 10, 5)

            selected_modes = st.multiselect(
                "Search mechanisms",
                list(SEARCH_MODES.keys()),
                default=list(SEARCH_MODES.keys())
            )

            compare = st.form_submit_button(
                "Run comparison",
                type="primary"
            )

        if compare:

            st.session_state.pop("lab_results", None)

            if not query.strip() or not selected_modes:

                st.warning(
                    "Enter a query and select search modes."
                )

            else:

                summaries = []
                all_results = {}

                with st.spinner("Running comparison..."):

                    for label in selected_modes:

                        mode = SEARCH_MODES[label]

                        start = time.perf_counter()

                        results = engine.search(
                            mode=mode,
                            text=query,
                            limit=top_k
                        )

                        latency = (
                            time.perf_counter() - start
                        ) * 1000

                        summaries.append({
                            "Search mechanism": label,
                            "Latency (ms)": round(latency, 2),
                            "Results": len(results)
                        })

                        all_results[label] = {
                            "mode": mode,
                            "products": results
                        }

                st.session_state["lab_results"] = {
                    "summary": summaries,
                    "results": all_results,
                    "query": query
                }

        comparison = st.session_state.get("lab_results")

        if comparison:

            st.subheader("Performance comparison")

            st.caption(
                f"Query: {comparison['query']}"
            )

            df = pd.DataFrame(comparison["summary"])

            st.dataframe(
                df,
                hide_index=True,
                width="stretch"
            )

            st.bar_chart(
                df.set_index("Search mechanism")[
                    "Latency (ms)"
                ],
                color="#111111",
            )

            st.caption(
                "Single-run latency measurements with an "
                "already loaded model. Not a formal benchmark."
            )

            st.subheader("Search results")

            results = comparison["results"]

            columns = st.columns(len(results), gap="medium")

            for column, (label, result) in zip(
                columns, results.items()
            ):

                with column:

                    st.markdown(f"**{label}**")

                    render_products(
                        result["products"],
                        score_mode=result["mode"],
                    )

    with tab2:

        st.subheader("Catalog and model information")

        product_count, category_count = engine.catalog.statistics()
        embeddings = engine.vector.embeddings

        c1, c2, c3 = st.columns(3)

        c1.metric("Products", product_count)
        c2.metric("Product types", category_count)
        c3.metric("Indexed vectors", len(embeddings))

        c4, c5, c6 = st.columns(3)

        c4.metric(
            "Embedding dimensions",
            embeddings.shape[1]
        )

        c5.metric(
            "Vector memory",
            f"{embeddings.nbytes / 1024**2:.2f} MiB"
        )

        c6.metric("Compute device", "CPU")

        st.divider()

        st.markdown("### Technology stack")

        st.table({
            "Component": [
                "ML model",
                "Database",
                "Keyword search",
                "Fuzzy search",
                "Vector search",
                "Hybrid ranking",
                "Multimodal search",
                "Frontend"
            ],
            "Technology": [
                engine.vector.model_name,
                "SQLite",
                "FTS5 / BM25",
                "RapidFuzz",
                "NumPy / Exact brute-force",
                "Reciprocal Rank Fusion",
                "Weighted embedding fusion",
                "Streamlit"
            ]
        })

        st.markdown("### Search architecture")

        st.code(
            "User Interface\n"
            "      |\n"
            "Unified Search Engine\n"
            "      |\n"
            "Keyword / Fuzzy / Vector Search\n"
            "      |\n"
            "Ranking and Fusion\n"
            "      |\n"
            "Product Results",
            language="text"
        )
