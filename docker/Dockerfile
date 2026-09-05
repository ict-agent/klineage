ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG CODEX_CLI_VERSION=0.149.0
ARG KLINEAGE_UID=1000
ARG KLINEAGE_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        bubblewrap \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        jq \
        ninja-build \
        nodejs \
        npm \
        pkg-config \
        ripgrep \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN npm install --global "@openai/codex@${CODEX_CLI_VERSION}" \
    && npm cache clean --force \
    && codex --version \
    && nvcc --version

RUN set -eux; \
    if getent group "${KLINEAGE_GID}" >/dev/null; then \
        existing_group="$(getent group "${KLINEAGE_GID}" | cut -d: -f1)"; \
        groupmod --new-name klineage "${existing_group}"; \
    else \
        groupadd --gid "${KLINEAGE_GID}" klineage; \
    fi; \
    if getent passwd "${KLINEAGE_UID}" >/dev/null; then \
        existing_user="$(getent passwd "${KLINEAGE_UID}" | cut -d: -f1)"; \
        usermod \
            --login klineage \
            --home /home/klineage \
            --move-home \
            --gid "${KLINEAGE_GID}" \
            --shell /bin/bash \
            "${existing_user}"; \
    else \
        useradd \
            --uid "${KLINEAGE_UID}" \
            --gid "${KLINEAGE_GID}" \
            --create-home \
            --home-dir /home/klineage \
            --shell /bin/bash \
            klineage; \
    fi

RUN uv venv \
        --python /usr/bin/python \
        --system-site-packages \
        /opt/klineage/venv \
    && mkdir -p \
        /home/klineage/.cache/uv \
        /home/klineage/.codex \
        /workspace/klineage \
    && chown -R \
        "${KLINEAGE_UID}:${KLINEAGE_GID}" \
        /home/klineage \
        /opt/klineage \
        /workspace/klineage

# Only the small FlashInfer surface used by the evaluator is installed.  The
# base image already supplies PyTorch/CUDA; --no-deps avoids replacing them.
RUN uv pip install --python /opt/klineage/venv/bin/python \
        apache-tvm-ffi==0.1.13.post3 \
        cuda-pathfinder==1.8.1 \
        cupti-python==13.3.1 \
        einops==0.8.2 \
        nvidia-cuda-cupti==13.3.75 \
        nvidia-ml-py==13.580.82 \
        pyelftools==0.33 \
        tabulate==0.10.0 \
        tqdm==4.70.0 \
    && uv pip install --python /opt/klineage/venv/bin/python \
        --no-deps flashinfer-python==0.6.18

COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/klineage-entrypoint

ENV DEBIAN_FRONTEND= \
    HOME=/home/klineage \
    VIRTUAL_ENV=/opt/klineage/venv \
    PATH=/opt/klineage/venv/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    UV_CACHE_DIR=/home/klineage/.cache/uv \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    KLINEAGE_WORKSPACE=/workspace/klineage

USER klineage
WORKDIR /workspace/klineage

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/klineage-entrypoint"]
CMD ["bash"]
