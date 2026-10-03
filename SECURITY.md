# Security policy

## Supported versions

Only the latest commit on the maintained default branch is considered for security fixes during prototype development. No production support or service-level agreement is offered.

## Report privately

Do **not** open a public issue for vulnerabilities, exposed credentials, personal data, unsafe crisis behavior, prompt-injection bypasses, malicious media samples, or dependency/weight compromise.

Use the repository host's private vulnerability-reporting feature when enabled, or contact the repository owner through a private channel. If neither exists, send a minimal private request for a secure reporting channel; do not attach sensitive evidence yet.

Include:

- affected commit/version and configuration;
- impact and realistic attack path;
- minimal reproduction using synthetic data;
- whether camera, microphone, transcripts, raw media, logs, weights, or robot output are involved;
- suggested mitigation if known.

Do not include real user data. Encrypt sensitive attachments only after agreeing on a channel.

## Response process

Maintainers should acknowledge a report, assign severity/owner, reproduce safely, preserve minimal evidence, prepare tests and a fix, review downstream impact, and coordinate disclosure. Timelines depend on severity and maintainer availability; this prototype makes no guaranteed response time. A critical privacy or physical-safety issue should disable the affected feature until reviewed.

## Security boundaries

The default configuration is for loopback/local research. It does not provide deployment-grade authentication, multitenant isolation, encrypted persistence, public rate limiting, a web application firewall, or guaranteed media sandboxing. In-memory session IDs are bearer-like capabilities and must not be exposed.

Never deploy publicly without:

- TLS, authentication/authorization, CSRF/CORS review, rate limits, request and worker resource ceilings;
- isolated media decoding and model workers;
- secret management, patching, monitoring, backups/retention policy, incident response, and deletion verification;
- signed/checksummed model artifacts and dependency provenance;
- a deployment-specific privacy and threat assessment;
- operator-controlled robot emergency stop where applicable.

## Secrets and sensitive data

Use environment variables for optional secrets and keep `.env` untracked. Logs must not contain raw media, full transcripts, face embeddings, presentation estimates, credentials, authorization headers, or query/body dumps. Revoke any exposed credential; removing it from Git history is not enough.

## Safe research

Do not bypass authentication or dataset terms. Do not upload exploit samples containing personal information. Avoid testing against systems or people without authorization. Coordinate before publishing an unpatched vulnerability.

See [docs/threat_model.md](docs/threat_model.md) for the current threat analysis.
