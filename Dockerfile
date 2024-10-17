FROM nvcr.io/nvidia/pytorch:24.06-py3

# Accept build arguments for user/group IDs
ARG USER_ID
ARG GROUP_ID
ARG USER_NAME=userapp

# Install system-wide packages
RUN pip install huggingface_hub plotly datasets nvidia-ml-py3 transformers
RUN pip install -U kaleido

# Create Triton directory
RUN mkdir -p /root/.triton/autotune

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