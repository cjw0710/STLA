FROM m.daocloud.io/docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 基础工具
RUN apt-get update && apt-get install -y --no-install-recommends \
      apt-utils build-essential git curl ca-certificates cmake libomp-dev tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata

# 安装 uv 到 /usr/local/bin，并立刻校验
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /usr/local/bin/uv --version && \
    ls -l /usr/local/bin | grep uv || true && \
    ls -l /root/.local/bin || true

# 不必额外改 PATH；若你想双保险也可以：
# ENV PATH="/usr/local/bin:/root/.local/bin:${PATH}"

# 使用国内镜像：注意 uv 调用的是 pip，建议设置标准 pip 环境变量
ENV PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
# 安装 CUDA 11.7 对应的 PyTorch 源（非常关键，否则会装到 CPU 轮子）
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu117"

# 安装 PyTorch（指定版本，避免被升级到不匹配 CUDA 的版本）
RUN /usr/local/bin/uv pip install --system --no-cache-dir \
    torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117

# 其他依赖
RUN /usr/local/bin/uv pip install --system --no-cache-dir --no-build-isolation \
    pandas scipy torch_geometric torch_sparse torch_scatter einops matplotlib pipreqs

# 可选：微软字体（修正了末尾反斜杠）
RUN apt-get update && \
    apt-get install -y software-properties-common debconf-utils && \
    apt-add-repository multiverse && \
    apt-get update && \
    echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections && \
    apt-get install -y ttf-mscorefonts-installer fontconfig && \
    fc-cache -fv

WORKDIR /workspace