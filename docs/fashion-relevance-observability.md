# Fashion relevance observability

Set `FASHION_LENS_RELEVANCE_MODE=SHADOW` to observe zero-shot classifications while keeping image search results unchanged. The app emits one JSON line to stdout per classified image query that reaches the search engine. Cloud Run parses the line into `jsonPayload`. [Cloud Run logging](https://cloud.google.com/run/docs/logging)

Each prepared query has a random UUID. Changing Discovery filters, rerunning Streamlit, or confirming the same prepared image reuses that identifier and does not emit a second event. Submitting another query starts a new observation. The event records the first search attempt. In SOFT mode a confirmation prompt sets `search_status` to `awaiting_confirmation`, with null search latency and result count. An explicit override or removal emits a separate `fashion_relevance_action` event with the same query ID, action, search status, latency and result count. Ordinary reruns do not repeat the action. OFF and standalone Engineering Lab assessments produce no search event.

- `would_reject` is the classifier's non-fashion flag, independent of policy and search mode. It is null when classification fails.
- `actually_rejected` is true when relevance policy blocks a search. It is false for a confirmation prompt and for all SHADOW observations.
- `classification_status=failed` distinguishes classifier failure from valid fashion and non-fashion results. `error_code` distinguishes initialization from scoring failures. Technical classifier failures allow retrieval in every mode. `search_status=failed` describes a separate retrieval failure.
- Classification latency measures raw category scoring and excludes the first-time prompt cache setup. Search latency measures the retrieval stage, including text inference and fusion for multimodal queries. Image preparation and eligibility filtering are outside this timer. For an image-removal action, search latency measures the text-only engine call. A completed search can have zero results.
- The application version comes from the project metadata, and `application_revision` identifies the Cloud Run revision when available. Model version identifies the loaded CLIP checkpoint and includes the revision when available. Prompt version is the SHA-256 digest of the validated configuration. Classifier version identifies the scoring implementation and prompt configuration digest. Category scores and margin are raw cosine values.
- Events omit images, embeddings, filenames, raw query text and user identifiers. Logging is best effort and scoped to the process. Events may be lost if a sink fails or the process stops. An observation has no ground-truth label and cannot establish accuracy or product relevance.

## Logs Explorer

Use the following base filter and append one of the filters below. Replace the service name if your Cloud Run service uses a different name. [Logging query language](https://cloud.google.com/logging/docs/view/logging-query-language)

```text
resource.type="cloud_run_revision"
resource.labels.service_name="fashionlens"
log_id("run.googleapis.com/stdout")
jsonPayload.event="fashion_relevance_evaluation"
```

Find SHADOW searches the classifier flagged that still completed.

```text
jsonPayload.relevance_mode="SHADOW"
jsonPayload.would_reject=true
jsonPayload.actually_rejected=false
jsonPayload.search_status="completed"
```

Find classifier failures where retrieval still completed.

```text
jsonPayload.classification_status="failed"
jsonPayload.search_status="completed"
```

Find slow completed multimodal retrievals.

```text
jsonPayload.search_mode="multimodal"
jsonPayload.search_status="completed"
jsonPayload.search_latency_ms>=250
```

Correlate by anonymous ID or filter to a specific classifier version.

```text
jsonPayload.query_id="b6881fef4c5a46ba92ed5427a4cf4e09"
```

```text
jsonPayload.classifier_version="clip-zero-shot-v1:REPLACE_WITH_CONFIG_DIGEST"
```
