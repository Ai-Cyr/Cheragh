# Production HTTP server

The FastAPI application has conservative process-level boundaries, but it is
still intended to run behind a TLS-terminating reverse proxy or API gateway.

## Secure startup

Set the secret through the process environment or a secret manager, not as a
literal command-line argument:

```bash
export CHERAGH_API_KEY="$(your-secret-manager read cheragh-api-key)"
export CHERAGH_REQUIRE_AUTH=true
cheragh serve --config rag.yaml --host 0.0.0.0
```

`cheragh serve` automatically requires authentication for non-loopback binds.
`CHERAGH_REQUIRE_AUTH=true` also makes application startup fail when no usable
key is configured. `/health` and `/ready` remain unauthenticated so an
orchestrator can probe the process without receiving corpus or engine data.
An injected `readiness_check` must be fast and non-blocking; use it to inspect
already-maintained dependency state, not to issue a fresh provider request.

Protected routes use `X-API-Key`. Comparisons are constant-time, failures do
not distinguish missing and incorrect keys, and prompts are not returned by
default. Programmatic deployments must explicitly set
`allow_prompt_exposure=True` before `include_prompt` can be used.

## Process boundaries

`create_app()` exposes the following controls:

| Control | Default | Behaviour |
| --- | ---: | --- |
| `max_request_body_bytes` | 1 MiB | Rejects declared and streamed oversize bodies with `413`. |
| `max_concurrent_operations` | 16 | Shared bound for ask, stream and index work; saturation returns `503`. |
| `max_server_connections` | 128 | Uvicorn ceiling; must exceed the operation bound so health probes retain headroom. |
| `request_timeout_seconds` | 60 s | Returns `504` when a synchronous ask exceeds the deadline. |
| `index_timeout_seconds` | 900 s | Returns `504` when indexing exceeds the deadline. |
| `stream_max_duration_seconds` | 300 s | Stops a stream between chunks after its deadline. |
| `max_top_k` | 50 | Rejects larger retrieval requests with `422`. |

`cheragh serve` exposes these controls with matching dashed options, for
example `--max-concurrent-operations 8 --request-timeout-seconds 30`. Use
`--require-auth` to enforce authentication locally as well. `--no-require-auth`
is an explicit override and must not be used on a public interface.

A timed-out synchronous operation cannot be forcefully killed safely. It keeps
occupying its concurrency permit until the worker really exits, preventing a
series of timeouts from creating an unbounded hidden worker backlog.

Every HTTP response has `X-Request-ID`, `Cache-Control: no-store`, and
`X-Content-Type-Options: nosniff`. A caller-supplied request ID is retained only
when it contains a bounded, log-safe identifier. Completion logs contain the
request ID, method, path, status, and duration; request and response bodies are
not logged. The same identifier is available to downstream FastAPI code as
`request.state.request_id`. Unexpected exceptions are recorded without their
possibly sensitive message and returned as a generic `500` response with a
correlation ID.

## Indexing endpoint

`POST /index` is disabled by default. Enabling it requires all three settings:

```bash
export CHERAGH_ENABLE_INDEXING=true
export CHERAGH_INDEX_ROOT=/srv/cheragh/indexing
export CHERAGH_API_KEY="$(your-secret-manager read cheragh-index-key)"
```

The root must be an existing directory. Both input and output paths are
resolved beneath it, including symlink resolution. Invalid paths and internal
indexing exceptions are redacted from HTTP responses. Prefer a separate,
non-public indexing deployment when runtime ingestion is not essential.

## Infrastructure still required

The application does not replace infrastructure controls. Configure the edge
proxy for TLS, request/header timeouts, header-size limits, IP or identity rate
limits, and an overall streaming deadline. The built-in API key is a single
shared-secret boundary; use a gateway with OIDC/mTLS and propagate a verified
principal when individual identity or tenant authorization is required.

Concurrency limits are per worker, not cluster-wide. A stream deadline is
checked between generated chunks and cannot interrupt a provider call blocked
inside `next()`. Provider SDK timeouts and circuit breakers must therefore also
be configured at the LLM, embedding, and vector-store clients.
