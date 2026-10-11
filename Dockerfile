# Ubuntu 24.04 LTS (Python 3.12)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV HF_HOME="/workspace/hf_cache"

# Install core utilities, the SSH server, and Jupyter UI dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    curl nano git numactl ca-certificates \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Configure the SSH daemon to accept RunPod's automated key injection
RUN mkdir /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

WORKDIR /workspace

ENV VIRTUAL_ENV=/workspace/heretic_env
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install the machine learning stack alongside JupyterLab
RUN pip install --upgrade pip && \
    pip install \
    torch "jax[cuda12]" git+https://github.com/google/torchax.git@main \
    transformers accelerate huggingface_hub fastapi uvicorn pydantic jupyterlab \
    --extra-index-url https://download.pytorch.org/whl/cu124

RUN cat << 'EOF' > /workspace/main_heretic.py
import os
import time

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HOME"] = "/workspace/hf_cache"

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax.dlpack as jx_dlpack
import torch
import torch.utils.dlpack as th_dlpack
import torchax

try:
    from torchax.interop import torch_view
    wrap_jax = torch_view
except ImportError:
    from torchax.tensor import Tensor as wrap_jax

from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import uvicorn

# Security Layer: Fetches your custom token from RunPod's Environment Variables
API_KEY = os.environ.get("API_AUTH_TOKEN", "default_failsafe_token")
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized Access")
    return api_key

class HereticSampler:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.canvas_size = 256

    def denoising_step(self, canvas):
        with torch.no_grad():
            outputs = self.model(input_ids=canvas, output_hidden_states=False)
        return outputs.logits

    def generate_canvas(self, prompt, num_steps=15, confidence_threshold=0.8):
        start_time = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids = inputs["input_ids"]
        prompt_len = prompt_ids.shape[1]
        
        if prompt_len > self.canvas_size:
            raise ValueError(f"Prompt length exceeds canvas size.")
            
        canvas = torch.randint(0, self.tokenizer.vocab_size, (1, self.canvas_size))
        canvas[0, :prompt_len] = prompt_ids[0]
        
        for t in range(num_steps):
            logits = self.denoising_step(canvas)
            probs = torch.softmax(logits, dim=-1)
            confidence, _ = torch.max(probs, dim=-1)
            
            gen_confidence = confidence[0, prompt_len:]
            if torch.all(gen_confidence > 0.95):
                break
                
            noise_mask = confidence < confidence_threshold
            noise_mask[0, :prompt_len] = False
            
            random_tokens = torch.randint(0, self.tokenizer.vocab_size, canvas.shape)
            best_tokens = torch.argmax(logits, dim=-1)
            canvas = torch.where(noise_mask, random_tokens, best_tokens)
            
        generated_ids = canvas[0, prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

app = FastAPI(title="Hackable Heretic dLLM (Secure Remote)")
MODEL_ID = "edwixx/diffusiongemma-26B-A4B-it-HERETIC-Uncensored"

print("1. Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("2. Staging PyTorch weights in System RAM...")
model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)

print("3. Executing Hybrid AI Power Bridge to JAX Mesh...")
torchax.enable_globally()
mesh = Mesh(jax.devices(), axis_names=('model',))
sharding = NamedSharding(mesh, P('model'))

for name, param in model.named_parameters():
    cpu_capsule = th_dlpack.to_dlpack(param.detach())
    jax_tensor = jx_dlpack.from_dlpack(cpu_capsule)
    jax_sharded = jax.device_put(jax_tensor, sharding)
    param.data = wrap_jax(jax_sharded)

sampler = HereticSampler(model, tokenizer)

class ChatRequest(BaseModel):
    prompt: str
    steps: int = 20
    confidence: float = 0.85

# The endpoint is now cryptographically locked to your authentication token
@app.post("/v1/generate", dependencies=[Depends(verify_key)])
async def generate(req: ChatRequest):
    return {"response": sampler.generate_canvas(req.prompt, req.steps, req.confidence)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Create the Multi-Service Boot Script
RUN cat << 'EOF' > /workspace/start.sh
#!/bin/bash
# 1. Start the SSH Daemon for secure terminal access
/usr/sbin/sshd

# 2. Start JupyterLab (Web Interface) using your custom token for security
jupyter-lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token="$API_AUTH_TOKEN" &

# 3. Start the Heretic Engine
numactl --interleave=all python /workspace/main_heretic.py
EOF

RUN chmod +x /workspace/start.sh

# Expose required ports (SSH, API, Jupyter)
EXPOSE 22 8000 8888
ENTRYPOINT []
CMD ["/workspace/start.sh"]
