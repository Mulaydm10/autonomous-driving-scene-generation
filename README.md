<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:000000,50:0a0a2e,100:001a3d&height=220&text=Autonomous%20Driving%20Scene%20Generation&fontSize=32&fontColor=00d4ff&fontAlignY=45&desc=Yandex%20Cup%20ML%20%E2%80%A2%20Controllable%20Video%20Synthesis%20via%20Diffusion&descSize=15&descAlignY=68&descFontColor=7ecfff&animation=blinking" width="100%"/>

<br/>

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+cu121-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.33.1-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/docs/diffusers)
[![Docker](https://img.shields.io/badge/Docker-Submission-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![DVC](https://img.shields.io/badge/DVC-Model%20Versioning-945DD6?style=for-the-badge)](https://dvc.org/)
[![Competition](https://img.shields.io/badge/Yandex%20Cup%20ML-2024-FF0000?style=for-the-badge&logo=yandex&logoColor=white)](https://yandex.com/cup/ml/)

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=14&pause=1200&color=00D4FF&center=true&vCenter=true&width=600&height=40&lines=CogVideoX-2b+%7C+STDiT3-XL%2F2+%7C+Flash+Attention+2;Camera+Conditioning+%7C+Bounding+Box+Embedding;Controllable+%E2%80%A2+Photorealistic+%E2%80%A2+Scene-Aware" alt="Typing SVG" />

</div>

---

## The Problem

Training data for autonomous driving systems requires thousands of hours of real-world footage across diverse conditions — rare scenarios, edge cases, adverse weather. Collecting it is expensive and dangerous. **This project generates it synthetically.**

Given a text description, a camera angle, and bounding boxes for road objects, the model synthesises photorealistic driving scene videos that autonomous systems can train on — no dashcam required.

---

## Architecture

Built on **CogVideoX-2b** with a custom **STDiT3-XL/2** backbone modified for controllable generation:

```
┌─────────────────────────────────────────────────────┐
│                    INPUT CONDITIONING               │
│  Text Prompt   Camera Angles   Bounding Boxes       │
│  "rainy night" [pitch, yaw, roll] [cars, pedestrians│
│   highway..."  per frame        trucks, signs...]   │
└────────────┬────────────┬──────────────┬────────────┘
             │            │              │
             ▼            ▼              ▼
┌─────────────────────────────────────────────────────┐
│              STDiT3-XL/2 (Custom)                   │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Text Cross │  │ Camera Angle │  │   Bbox    │  │
│  │  Attention  │  │  Embedding   │  │ Embedding │  │
│  └──────┬──────┘  └──────┬───────┘  └─────┬─────┘  │
│         └────────────────┴────────────────┘         │
│                          │                          │
│              ┌───────────▼────────────┐             │
│              │  Temporal Self-Attn    │             │
│              │  (Flash Attention 2)   │             │
│              └───────────┬────────────┘             │
└──────────────────────────┼──────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   CogVideoX-2b Decoder  │
              │   3D VAE → Video frames │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  OUTPUT: Driving video  │
              │  Controllable scene     │
              └─────────────────────────┘
```

**Bounding box classes:** `car` · `truck` · `bus` · `person` · `traffic_sign`

---

## Key Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Base model | CogVideoX-2b | Strong temporal coherence, open weights |
| Backbone | STDiT3-XL/2 | Scalable DiT with spatial-temporal factorisation |
| Camera conditioning | Per-frame angle embedding (pitch/yaw/roll) | Enables viewpoint-consistent generation |
| Bbox conditioning | Learned token embedding per object class | Spatial control without pixel-level masks |
| Attention | Flash Attention 2.8.3 + xformers | 3–4× VRAM reduction vs vanilla attention |
| Positional encoding | Rotary embeddings (`rotary-embedding-torch`) | Better extrapolation to unseen sequence lengths |
| Submission | Docker + DVC | Reproducible builds; fast weight upload via S3 |

---

## Stack

```txt
torch==2.5.1+cu121        # CUDA 12.1 backend
diffusers==0.33.1          # Pipeline abstractions
transformers==4.44.2       # Text encoder
acccelerate==1.6.0         # Multi-GPU / mixed precision
flash-attn==2.8.3          # Memory-efficient attention
xformers                   # Additional attention kernels
mmengine>=0.10.3           # Config + registry system
timm==0.9.16               # Vision backbone utilities
rotary-embedding-torch     # RoPE positional encoding
einops                     # Tensor reshaping
```

---

## Run Inference

```bash
# Build the Docker image (matches competition environment)
docker build -t driving-gen .

# Run generation
docker run --rm --gpus="all" --runtime=nvidia \
  -v /path/to/dataset:/workspace/input_data \
  -v /path/to/output:/workspace/output_data:rw \
  driving-gen
```

Or directly:

```bash
pip install -r requirements.txt
python inference.py --config config.py
```

---

## Repo Layout

```
.
├── src/
│   ├── models/        # STDiT3-XL/2 with custom conditioning heads
│   ├── data/          # Dataset loaders + bbox preprocessing
│   ├── schedulers/    # Diffusion noise schedules
│   └── utils/         # Visualisation + metric helpers
├── inference.py       # Entry point
├── config.py          # Model + generation config
├── Dockerfile         # Competition submission container
├── init_dvc.sh        # DVC remote setup (model weight versioning)
└── requirements.txt
```

---

## Competition

Built for **Yandex Cup ML 2024** — an international ML competition hosted by Yandex. Task: controllable video generation for autonomous driving scenarios with camera angle and object class conditioning.

---

<div align="center">
<sub>Built by <a href="https://github.com/Mulaydm10">Dhruv Mulay</a> · PyTorch · CogVideoX · Flash Attention</sub>
</div>
