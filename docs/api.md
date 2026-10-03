# API guide

## Start locally

Install the API dependencies and run on loopback:

```powershell
python -m pip install -e ".[api,dev]"
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Interactive docs are at `/docs` and `/redoc`. Responses include `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, and an `X-Request-ID`.

## Lifecycle

1. Create one session with exact consent.
2. Send only modalities covered by that consent.
3. Analyze one or more signals.
4. Optionally request a dialogue response using the last analysis.
5. Delete the session. If not deleted, it expires after the configured TTL.

Session IDs are bearer-like capabilities. Keep them out of logs, screenshots, and shared transcripts. This prototype has no user authentication.

## Endpoints

### `GET /health`

Returns version, mock-capable mode, and active session count. It does not prove that optional models, sensors, or downstream resources are healthy.

### `POST /session`

Creates an in-memory session and returns HTTP 201.

```json
{
  "consent": {
    "microphone": false,
    "camera": false,
    "text_analysis": true,
    "age_band_analysis": false,
    "perceived_gender_presentation_analysis": false,
    "store_raw_data": false,
    "cloud_upload": false,
    "store_optional_attributes": false
  },
  "preferences": {
    "preferred_name": null,
    "preferred_form_of_address": null,
    "self_reported_age_band": null,
    "reading_level": "standard",
    "response_pace": "normal"
  },
  "mock_mode": true
}
```

Age/presentation analysis requires camera consent. Attribute storage requires the corresponding analysis permission. Cloud upload is rejected by this local-only build.

### `DELETE /session/{session_id}`

Clears application-held in-memory state and returns `{"deleted": true}`. A missing or already deleted session returns a structured error.

### `POST /analyze/text`

Requires `text_analysis` consent. Body:

```json
{
  "session_id": "replace-with-live-session-id",
  "text": "I feel a bit overwhelmed by the deadline."
}
```

Text length is 1–10,000 characters. The current analyzer is a lightweight local heuristic. Safety routing is separate from affect fusion.

### `POST /analyze/audio`

Requires `microphone` consent. Send `multipart/form-data` fields:

- `session_id`: live session ID;
- `file`: a non-empty PCM WAV named `.wav` or `.wave`.

Accepted API content types are `audio/wav`, `audio/x-wav`, and `audio/wave`. The analyzer verifies extension and RIFF/WAVE signature. Default maximum media size is 10 MiB. Decoding, quality assessment, feature extraction, and the bundled conservative baseline are experimental and unvalidated. The shipped configurations use that baseline; a CNN or CRNN runs only when an operator explicitly selects a matching, reviewed local checkpoint in the `audio` configuration. Architecture availability does not mean a model is trained or enabled.

```powershell
curl.exe -X POST http://127.0.0.1:8000/analyze/audio -F "session_id=YOUR_SESSION_ID" -F "file=@sample.wav;type=audio/wav"
```

### `POST /analyze/image`

Requires `camera` consent. Send `multipart/form-data` fields `session_id` and `file`. Accepted types are JPEG and PNG, with signature/decoder validation in the analyzer. The service abstains for inadequate quality, no face, or ambiguous multiple faces rather than silently choosing a person.

Optional attributes require their own consent. They are disabled by default, reported separately, and never enter fusion. No validated optional-attribute weights ship with the repository.

### `POST /analyze/multimodal`

The runnable deterministic path accepts text and/or mock audio/vision signals:

```json
{
  "session_id": "replace-with-live-session-id",
  "text": "I could use a quiet moment.",
  "mock_audio": {
    "label": "sad",
    "confidence": 0.78,
    "quality": 0.9,
    "temporal_consistency": 0.85
  },
  "mock_vision": {
    "label": "happy",
    "confidence": 0.75,
    "quality": 0.9,
    "temporal_consistency": 0.9,
    "face_count": 1
  }
}
```

At least one modality is required. Each supplied modality requires matching session consent. A `face_count` above one returns ambiguity; zero returns no-face abstention. Strong contradictory signals should produce an uncertain fused result.

Mock signals are test controls supplied by the caller. They are not sensor inference and their confidence is not empirical.

### `POST /dialogue/respond`

Uses the request analysis or the session's last analysis. Example:

```json
{
  "session_id": "replace-with-live-session-id",
  "analysis": null,
  "user_message": "That estimate is wrong.",
  "estimate_is_incorrect": true,
  "advice_permission": false
}
```

`estimate_is_incorrect` suppresses the estimated emotion and acknowledges the correction. `advice_permission` is required before a grounding suggestion. User message safety routing runs before ordinary dialogue. Transcripts are untrusted data, not system instructions.

### `GET /privacy/settings`

Pass the live ID in the `X-Session-ID` header.

### `PUT /privacy/settings`

Pass `X-Session-ID` and a complete body:

```json
{
  "store_raw_data": false,
  "store_derived_features": false,
  "store_optional_attributes": false,
  "cloud_upload": false,
  "session_ttl_seconds": 1800
}
```

The schema bounds TTL to 60–86,400 seconds. Settings cannot exceed original consent. The core manager has no raw-media persistence implementation even if raw storage consent was requested; do not interpret a true flag as proof that durable storage exists.

## Analysis response

```json
{
  "session_id": "string",
  "timestamp": "2026-01-01T00:00:00Z",
  "modalities_available": ["audio", "vision", "text"],
  "modality_results": {
    "audio": {
      "label": "uncertain",
      "confidence": 0.2,
      "quality": 0.8,
      "probabilities": {},
      "available": true,
      "temporal_consistency": 1.0,
      "reason": "string",
      "metadata": {}
    }
  },
  "fused_affect": {
    "label": "uncertain",
    "confidence": 0.2,
    "uncertain": true,
    "probabilities": {},
    "contributions": {"audio": 1.0},
    "explanation": "The available signals are weak or ambiguous; this is not a medical assessment."
  },
  "optional_demographic_estimates": {
    "age_band": "analysis-disabled",
    "perceived_gender_presentation": "analysis-disabled",
    "confidence": 0.0,
    "uncertain": true,
    "notice": "Optional visual presentation estimates are disabled by default and are not emotion evidence."
  },
  "safety": {
    "route": "normal",
    "human_support_recommended": false,
    "message": null,
    "resources": []
  }
}
```

Exact fields are authoritative in the generated OpenAPI schema.

## Errors and limits

Errors use:

```json
{
  "error": {
    "code": "validation_error",
    "message": "invalid request fields: body.text",
    "request_id": "opaque-id"
  }
}
```

Common statuses are 400 invalid request, 403 consent required, 404 missing session, 410 expired session, 413 too large, 415 invalid media, 422 schema validation, 503 capacity, and 504 timeout. Validation errors list field locations without echoing submitted values.

Defaults: 10 MiB media limit, 30-second request timeout, 1,000 live sessions, and 30-minute TTL. Multipart framing has bounded overhead; the media read enforces the exact byte limit.

## Production warning

Do not expose this API to the internet as-is. It lacks authentication and deployment-grade isolation. Follow [threat_model.md](threat_model.md) and [SECURITY.md](../SECURITY.md).
