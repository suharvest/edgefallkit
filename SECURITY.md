# Security policy

## Supported versions

Security fixes currently target the latest published release candidate.

## Reporting a vulnerability

Do not publish credentials, private RTSP URLs, broker passwords, or device logs
containing sensitive data in a public issue. Report a vulnerability privately
to the project maintainers through the security contact configured on the
eventual source repository. A public contact address must be added before the
repository is announced broadly.

## Deployment guidance

- Use MQTT TLS and per-device credentials outside trusted development networks.
- Keep RTSP credentials in local configuration or secret stores, not Compose or
  Git.
- Pin runtime images by the published digest for production deployment.
- Review model and dataset terms before enabling automatic downloads.
- Treat fall alerts as decision support, not as a certified emergency system.
