# Model artifacts

The ONNX pose models and engines used in the Jetson benchmarks are backed up on
Spark rather than copied into this source project. The `.engine` files are
valid only for the compatible SM87 / TensorRT 10.3 environment. Prefer
rebuilding from ONNX on the deployment target.

See [`../ASSET_LOCATIONS.md`](../ASSET_LOCATIONS.md) for paths and rebuild
instructions. Review the upstream model license before distribution.
