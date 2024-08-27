FROM nvcr.io/nvidia/pytorch:24.06-py3

RUN pip install huggingface_hub plotly datasets nvidia-ml-py3
RUN pip install -U kaleido

RUN mkdir -p /root/.triton/autotune