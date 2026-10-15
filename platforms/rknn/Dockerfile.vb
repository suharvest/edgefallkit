ARG BASE=vb-base-rk:m21
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends python3-numpy \
    && rm -rf /var/lib/apt/lists/*

COPY platforms/rknn/fall_core.py /opt/vb/py/fall_core.py
COPY platforms/rknn/fall_app.py /opt/vb/py/fall_app.py

ENV PYTHONPATH=/opt/vb/py \
    LD_LIBRARY_PATH=/opt/vb/bin:/opt/onnxruntime/lib \
    VB_PRODUCTION=1

ENTRYPOINT ["python3", "-m", "vision_base.main"]
