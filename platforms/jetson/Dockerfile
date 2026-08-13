# Jetson Orin Nano/NX runtime.  The image deliberately contains no PyTorch,
# Ultralytics or ONNX Runtime.  TensorRT/CUDA are supplied by the target host
# at runtime (see docker-compose.yml). The build stage needs nvcc for the
# CUDA preprocess kernel; set this image to the matching JetPack/CUDA devel
# tag when building on a different Jetson release.
ARG L4T_CUDA_DEVEL_IMAGE=nvcr.io/nvidia/l4t-cuda:12.6.0-devel-r36.4.0
ARG L4T_RUNTIME_IMAGE=nvcr.io/nvidia/l4t-base:r36.4.0
FROM ${L4T_CUDA_DEVEL_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config libopencv-dev \
    libnvinfer-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src
RUN cmake -S . -B build -DBUILD_APP=ON -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j2 --target jetson_fall_trt

ARG L4T_RUNTIME_IMAGE=nvcr.io/nvidia/l4t-base:r36.4.0
FROM ${L4T_RUNTIME_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5d libopencv-imgproc4.5d libopencv-videoio4.5d \
    python3 python3-numpy python3-opencv python3-paho-mqtt \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/build/libjetson_fall_trt.so /app/libjetson_fall_trt.so
COPY app.py /app/app.py
COPY config/config.json /app/config/config.json
COPY config/config.schema.json /app/config/config.schema.json
ENTRYPOINT ["python3", "/app/app.py"]
