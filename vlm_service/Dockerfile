# VLM service image: PyTorch 2.14.1 with CUDA 13.0, the same torch build as the native setup.
# Built by docker-compose.yml. The model weights are not in the image: the compose file mounts the
# Hugging Face cache.
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
# torch and torchvision are already in the base image at the pinned versions, so pip keeps them. The base
# image installs them into Ubuntu's Python, which PEP 668 marks as "externally managed"; in a container,
# installing the other packages next to them is intended, hence --break-system-packages.
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY service.py backends.py ./
CMD ["python", "service.py"]
