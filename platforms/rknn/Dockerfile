ARG BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim-bookworm
FROM ${BASE_IMAGE} AS builder
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
RUN sed -i -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
           -e 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
           -e 's|http://security.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
           /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip wheel --no-cache-dir --wheel-dir /wheels numpy==1.26.4 opencv-python-headless==4.10.0.84 pybind11==2.13.6
COPY cpp/rknn_postprocess.cpp /build/rknn_postprocess.cpp
RUN pip install --no-cache-dir /wheels/pybind11-*.whl \
    && c++ -O3 -DNDEBUG -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
       /build/rknn_postprocess.cpp -o /build/rknn_postprocess$(python3-config --extension-suffix) \
    && strip --strip-unneeded /build/rknn_postprocess$(python3-config --extension-suffix)

FROM ${BASE_IMAGE} AS runtime
RUN sed -i -e 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' \
           -e 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
           -e 's|http://security.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' \
           /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
      gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/numpy-*.whl /wheels/opencv_python_headless-*.whl && rm -rf /wheels

WORKDIR /opt/fall-detection
COPY --from=builder /build/rknn_postprocess*.so ./
COPY app.py benchmark.py fall_core.py rknn_pose.py video_source.py runtime_smoke.py ./
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/usr/lib/python3/dist-packages
ENTRYPOINT ["python3"]
CMD ["app.py", "--config", "/config/config.json"]
