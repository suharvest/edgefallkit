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

## Official YOLOv8m-Pose HEF

Hailo Model Zoo v2.19.0 provides a Hailo-8 `yolov8m_pose.hef` with 3 contexts
and 9 raw outputs. It is 31,608,992 bytes and is not stored in git.

`https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolov8m_pose.hef`

SHA256: `fa0bfbf83dba494f4d75ec2fd0ef497ca9d402a65c324afc9865ffc327a53514`.
