ARG BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim-bookworm
ARG MIRROR_SED='s|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g; s|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g; s|http://security.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g'

FROM ${BASE_IMAGE} AS builder
ARG MIRROR_SED
RUN sed -i -e "${MIRROR_SED}" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip wheel --no-cache-dir --wheel-dir /wheels numpy==1.26.4 opencv-python-headless==4.10.0.84 pybind11==2.13.6
COPY cpp/rknn_postprocess.cpp /build/rknn_postprocess.cpp
RUN pip install --no-cache-dir /wheels/pybind11-*.whl \
    && c++ -O3 -DNDEBUG -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
       /build/rknn_postprocess.cpp -o /build/rknn_postprocess$(python3-config --extension-suffix) \
    && strip --strip-unneeded /build/rknn_postprocess$(python3-config --extension-suffix)

# Exists only to supply the GstVideo/GstRtsp typelibs and the appsink plugin.
# Installing gir1.2-gst-plugins-base-1.0 in the runtime stage would pull
# libgstreamer-gl1.0-0 -> libgl1 -> mesa -> libllvm15, about 150 MB of OpenGL
# stack for an app that only decodes video into an RGB array.
FROM ${BASE_IMAGE} AS gstgir
ARG MIRROR_SED
RUN sed -i -e "${MIRROR_SED}" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
       gir1.2-gst-plugins-base-1.0 gstreamer1.0-plugins-base \
    && rm -rf /var/lib/apt/lists/*

FROM ${BASE_IMAGE} AS runtime
ARG MIRROR_SED
RUN sed -i -e "${MIRROR_SED}" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3-gi gir1.2-gstreamer-1.0 \
      libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 gstreamer1.0-plugins-good \
    && rm -rf /var/lib/apt/lists/*
COPY --from=gstgir /usr/lib/aarch64-linux-gnu/girepository-1.0/GstVideo-1.0.typelib \
                   /usr/lib/aarch64-linux-gnu/girepository-1.0/GstRtsp-1.0.typelib \
                   /usr/lib/aarch64-linux-gnu/girepository-1.0/GstAudio-1.0.typelib \
                   /usr/lib/aarch64-linux-gnu/girepository-1.0/GstPbutils-1.0.typelib \
                   /usr/lib/aarch64-linux-gnu/girepository-1.0/GstTag-1.0.typelib \
                   /usr/lib/aarch64-linux-gnu/girepository-1.0/GstSdp-1.0.typelib \
                   /usr/lib/aarch64-linux-gnu/girepository-1.0/
COPY --from=gstgir /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstapp.so \
                   /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgsttypefindfunctions.so \
                   /usr/lib/aarch64-linux-gnu/gstreamer-1.0/
# The wheels are mounted rather than copied: a COPY layer keeps them in the
# image even after the next layer deletes them, costing 44 MB of pull forever.
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-cache-dir /wheels/numpy-*.whl /wheels/opencv_python_headless-*.whl

WORKDIR /opt/fall-detection
COPY --from=builder /build/rknn_postprocess*.so ./
COPY app.py benchmark.py fall_core.py rknn_pose.py video_source.py runtime_smoke.py ./
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/usr/lib/python3/dist-packages
ENTRYPOINT ["python3"]
CMD ["app.py", "--config", "/config/config.json"]
