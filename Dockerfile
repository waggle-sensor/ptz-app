FROM nvcr.io/nvidia/pytorch:24.06-py3

# 1. Copy code
WORKDIR /app
COPY . ./

# --- TEMPORARY WORKAROUND for local testing on proxied networks ---
# WARNING: This is insecure and should be removed for the final build.
ENV PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org"
ENV HUGGING_FACE_HUB_DISABLE_CERT_CHECK=1
# ----------------------------------------------------------------

# 2. Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy and run the OpenCV fix script
COPY opencv-fix.py /tmp/
RUN python /tmp/opencv-fix.py

# Verify the fix worked
RUN python -c "import cv2; print('OpenCV imported successfully')"


# 3. Create a local cache directory for the model
RUN mkdir -p /hf_cache/microsoft/Florence-2-base
# RUN mkdir -p /hf_cache/microsoft/Florence-2-large

# 4. Download the model files info /hf_cache/microsoft/Florence-2-large
RUN huggingface-cli download \
	microsoft/Florence-2-large \
	--repo-type model \
	--cache-dir /hf_cache \
	--local-dir /hf_cache/microsoft/Florence-2-large \
	--resume \
	--force  # <-- ensures we overwrite any existing files or resume a download

# 5. Download the model files info /hf_cache/microsoft/Florence-2-base
RUN huggingface-cli download \
	microsoft/Florence-2-base \
	--repo-type model \
	--cache-dir /hf_cache \
	--local-dir /hf_cache/microsoft/Florence-2-base \
	--resume \
	--force  # <-- ensures we overwrite any existing files or resume a download

# 6. Set the environment variables for offline mode
ENV HF_HOME=/hf_cache
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1




# ENV TRANSFORMERS_CACHE=/hf_cache/

# # Download the model during the build process
# RUN python -c "\
# import torch; \
# from transformers import AutoProcessor, AutoModelForCausalLM; \
# model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', torch_dtype=torch.float32, trust_remote_code=True); \
# processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)"
#
ENTRYPOINT ["python", "./main.py"]
