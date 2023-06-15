
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.08-py3

FROM ${BASE_IMAGE} as base

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf \
        autogen \
        clangd \
        gdb \
        git-lfs \
        libb64-dev \
        libz-dev \
        locales-all \
        mosh \
        python3-dev \
        rapidjson-dev \
        unzip \
        zstd \
        zip \
        zsh \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==1.12.1+cu116 -f \
                    https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-cache-dir --extra-index-url https://pypi.ngc.nvidia.com \
                    regex fire tritonclient[all] && \
    pip3 install --no-cache-dir transformers huggingface_hub tokenizers sentencepiece \
                    sacrebleu datasets tqdm omegaconf rouge_score && \
    pip3 install --no-cache-dir cmake==3.24.3

# backend build
ADD . /workspace/build/fastertransformer_backend
RUN mkdir -p /workspace/build/fastertransformer_backend/build
WORKDIR /workspace/build/fastertransformer_backend/build

RUN cmake \
      -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/opt/tritonserver \
      -D TRITON_COMMON_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_CORE_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_BACKEND_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      ..

RUN make -j"$(grep -c ^processor /proc/cpuinfo)" install

FROM base as runtime

WORKDIR /opt/tritonserver
COPY --from=base /opt/tritonserver/backends/fastertransformer/ \
                        backends/fastertransformer

RUN apt-get update && apt-get install -y --no-install-recommends \
                        openssh-server && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        openssh-server && \
    rm -rf /var/lib/apt/lists/*

ENV NCCL_LAUNCH_MODE=PARALLEL
RUN sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config \
    && mkdir /var/run/sshd -p
