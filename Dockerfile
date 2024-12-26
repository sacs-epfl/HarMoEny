FROM nvcr.io/nvidia/pytorch:24.12-py3
# was 24.06

# Accept build arguments for user/group IDs
ARG USER_ID
ARG GROUP_ID
ARG USER_NAME=userapp

# Install system dependencies for FastMoE
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install system-wide packages
RUN pip install \
    setuptools==75.6.0 \
    wheel==0.45.0 \
    huggingface_hub==0.26.1 \
    datasets==3.0.2 \
    nvidia-ml-py3==7.352.0  \
    transformers==4.46.0  \
    deepspeed==0.15.3  \
    vllm==0.6.5 \
    deepspeed-mii==0.3.1

RUN python3 -m pip install gurobipy==12.0.0

RUN pip install -U kaleido==0.2.1
RUN python3 -m pip install -v -U --no-build-isolation git+https://github.com/microsoft/tutel@c7559e8


# Clone and install FastMoE
RUN git clone https://github.com/laekov/fastmoe.git /workspace/fastmoe && \
    cd /workspace/fastmoe && \
    git checkout ee3c5615a5588c65273a512de82f19ca287e1bdf && \
    git submodule update --init --recursive && \
    pip install .

# Create Triton directory
RUN mkdir -p /.triton/autotune

RUN groupadd -f -g ${GROUP_ID} ${USER_NAME} || true && \
    id -u ${USER_NAME} >/dev/null 2>&1 || useradd -l -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME} && \
    install -d -m 0755 -o ${USER_NAME} -g ${USER_NAME} /home/${USER_NAME}

# Create directory for non-root pip installations
RUN mkdir -p /home/${USER_NAME}/.local && \
    chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.local

# Set environment variables
ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

# Switch to non-root user
USER ${USER_NAME}

# Set the working directory
WORKDIR /workspace

# Set the PYTHONPATH environment variable for FastMoE
#ENV PYTHONPATH="/workspace/fastmoe:${PYTHONPATH}"