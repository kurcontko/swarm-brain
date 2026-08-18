# Security policy

## Supported

Only the `main` branch is supported. The project is pre-1.0; there are no
maintained release lines.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub Security Advisories
("Report a vulnerability" on the repository's Security tab). Do not open a
public issue for an unpatched vulnerability.

You can expect an acknowledgement within a week. Coordinated disclosure is
appreciated; a 90-day window is a reasonable default.

## Design posture

The API is fail-closed by design: every data route requires a signed
short-lived bearer token with server-derived identity and capabilities; the
interactive OpenAPI surface and the hosted demo trigger are disabled unless an
operator enables them explicitly. An internal pre-publication review covering
accepted risks and deploy-time mitigations is kept privately; report anything
it missed through the process above.
