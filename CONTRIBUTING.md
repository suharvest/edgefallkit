# Contributing

Contributions are welcome. Keep platform behavior comparable and preserve the
separation between the readable control plane and the accelerated data plane.

## Before opening a change

1. Do not commit model binaries, datasets, generated engines, credentials, or
   device-specific SDKs.
2. Keep the MQTT output compatible with `contracts/mqtt-result.schema.json`.
3. Record the device, power mode, input, concurrency, latency scope, CPU, RSS,
   accelerator utilization, and image size for performance claims.
4. Never present pose mAP or a looping positive clip as fall-detection
   accuracy. Follow `evaluation/EVALUATION.md` for accuracy changes.
5. Preserve upstream license and provenance information for model-related
   changes.

## Verification

Run the repository checks before submitting:

```bash
./tools/verify_baseline.sh
```

Then run the platform-specific tests documented in its README. Hardware changes
should include raw evidence and a checksum under the appropriate `results/` or
`evaluation/reports/` directory.

By submitting a contribution, you agree that it is licensed under Apache-2.0
unless the file is clearly marked as third-party material under another
license.
