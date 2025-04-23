 # ✨ ELoRA: Efficient LoRA and KV Cache Management for Multi-LoRA LLMs ✨

  ## 🔍 Overview

  ELoRA is a supercharged 🚀 inference system that transforms Multi-LoRA LLM serving through smart management of LoRA adapters and KV caches based on their usage patterns. By cleverly tracking these dependencies in a unified caching framework, ELoRA dramatically boosts inference efficiency and slashes latency in real-world production environments!

  ## 🏆 Breakthrough Performance

  <div align="center">
  <b>🎯 ELoRA delivers mind-blowing performance improvements over existing systems:</b>
  </div>

  - ⚡ **59.7% reduction** in Time-To-First-Token (TTFT)
  - 🔥 **33.2% reduction** in Time-Per-Output-Token (TPOT)
  - 🚀 **1.9× higher** peak throughput

  ## 💡 Key Innovations

  - **🧠 Dependency-Aware Cache Manager**: Intelligently eliminates invalid KV caches by tracking usage dependencies between LoRAs and KV caches
  - **⚙️ Performance-Driven Cache Swapper**: Makes brilliant swap decisions based on a unified cost model that precisely calculates benefits to inference performance
  - **🏊‍♂️ Unified Caching Pool**: Dynamically juggles memory resources between LoRAs and KV caches based on what your workload actually needs in real-time

  ## 🌐 Broad Compatibility

  - **📚 Models**: Works seamlessly with Llama2-7B/13B/34B and other decoder-only transformer architectures
  - **💻 Hardware**: Runs efficiently across various accelerators (NPUs, GPUs)
  - **🛠️ Applications**: Perfect for chatbots, multi-language translation, personal assistants, and all your Multi-LoRA needs!

## 🛠️ Quick Start Guide

### 🐳 Setting Up Your Environment

Get up and running in minutes with our Docker-based setup:

```bash
# Launch container with GPU support
docker run --name yourname -d --gpus all -it --ipc=host \
  -v /your/path/here:/root -p 12345:22 nvcr.io/nvidia/pytorch:23.10-py3

# Clone the ELoRA repository
git clone https://github.com/GiggleWang/ELoRA.git

# Navigate to project directory
cd ELoRA

# Enable multi-LoRA capability
export VLLM_INSTALL_PUNICA_KERNELS=1

# Install ELoRA
pip install -e .
```

### 🚀 Running ELoRA

Fire up the API server and start making inference requests:

```bash
# Launch the API server
python /root/ELoRA/vllm/entrypoints/openai/api_server.py \
    --model your_model/ \
    --lora-module your_lora

# Make your first request!
curl -X POST http://0.0.0.0:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your_model_and_lora",
    "prompt": "tell me something about ELoRA",
    "elora_id": your_elora_id,
    "max_tokens": 100,
    "temperature": 0.0,
    "stream": false,
    "ignore_eos": true
  }'
```

Now you're ready to experience the lightning-fast ⚡ performance of ELoRA with your own models and LoRA adapters!
