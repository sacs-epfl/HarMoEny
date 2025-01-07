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

# Remove old nvml
RUN pip uninstall -y pynvml

# Install system-wide packages
RUN pip install \
    setuptools==75.6.0 \
    wheel==0.45.1 \
    optimum==1.23.3 \
    huggingface_hub==0.26.1 \
    datasets==3.0.2 \
    nvidia-ml-py3==7.352.0  \
    deepspeed==0.15.3 \
    gekko==1.2.1 \
    torchvision==0.20.1 
    # transformers==4.46.0  \
    # vllm==0.6.5 \
    # deepspeed-mii==0.3.1

RUN python3 -m pip install gurobipy==12.0.0

RUN pip install -U kaleido==0.2.1
RUN python3 -m pip install -v -U --no-build-isolation git+https://github.com/microsoft/tutel@c7559e8

# Clone and install FastMoE
RUN git clone https://github.com/laekov/fastmoe.git /workspace/fastmoe && \
    cd /workspace/fastmoe && \
    git checkout ee3c5615a5588c65273a512de82f19ca287e1bdf && \
    git submodule update --init --recursive && \
    pip install .

# Reinstall flash-attn
#RUN pip install flash_attn==2.7.2.post1 -U --force-reinstall

# RUN pip uninstall -y transformers 
# RUN pip install git+https://github.com/huggingface/transformers.git@v4.37-release

RUN pip uninstall -y flash_attn
RUN git clone https://github.com/Dao-AILab/flash-attention /workspace/flash-attn && \
    cd /workspace/flash-attn && \
    git checkout f86e3dd && \
    python setup.py install

# Install AutoAWQ
RUN git clone https://github.com/casper-hansen/AutoAWQ /workspace/autoawq && \
    cd /workspace/autoawq && \
    git checkout cbd6a75b065e94a3e530dfdbb8f3973f0d954ec0 && \
    pip install .

# Install AutoGPTQ
RUN git clone https://github.com/AutoGPTQ/AutoGPTQ /workspace/autogptq && \
    cd /workspace/autogptq && \
    git checkout 323950b && \
    python setup.py install
  #  DISABLE_QIGEN=1 pip3 install .

# Install Transformers
# RUN pip uninstall -y transformers
# RUN git clone https://github.com/huggingface/transformers /workspace/transformers && \
#     cd /workspace/transformers && \
#     git checkout 8e3e145 && \
#     pip install .

# Create Triton directory
RUN mkdir -p /.triton/autotune

RUN groupadd -f -g ${GROUP_ID} ${USER_NAME} || true && \
    id -u ${USER_NAME} >/dev/null 2>&1 || useradd -l -u ${USER_ID} -g ${GROUP_ID} ${USER_NAME} && \
    install -d -m 0755 -o ${USER_NAME} -g ${USER_NAME} /home/${USER_NAME}

# Create directory for non-root pip installations
RUN mkdir -p /home/${USER_NAME}/.local && \
    chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/.local

RUN chown -R ${USER_NAME}:${USER_NAME} /.triton /workspace

# Set environment variables
ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

# Switch to non-root user
USER ${USER_NAME}

# Set the working directory
WORKDIR /workspace

# Set the PYTHONPATH environment variable for FastMoE
#ENV PYTHONPATH="/workspace/fastmoe:${PYTHONPATH}"