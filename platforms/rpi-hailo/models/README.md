# HEF model

Run `../deploy.sh --accept-upstream-license` to download the official Hailo Model Zoo
`yolov8s_pose.hef` for Hailo-8. The 10.1 MB binary is intentionally not stored
in git. The fixed upstream URL is:

`https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s_pose.hef`

The script rejects anything whose SHA256 is not
`e19856699ed47cf866d23265827f960b263f287dab5e54e82c7ce37e12525a2d`.
The HEF is mounted from `./models` by Compose and is not baked into the final
runtime image. The flag acknowledges upstream terms applicable to the model;
it does not grant redistribution rights. For an offline deployment use
`../deploy.sh --accept-upstream-license --offline --hef /path/to/yolov8s_pose.hef`.
The legacy downloader is a model-only wrapper and now also requires explicit
acceptance: `../scripts/fetch_model.sh --accept-upstream-license`.
