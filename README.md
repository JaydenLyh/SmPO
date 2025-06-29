# Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences (ICML 2025)

[arXiv Paper](https://arxiv.org/abs/2506.02698) | [Project Page](https://jaydenlyh.github.io/SmPO-project-page/) | [DDIM-InPO Project Page](https://jaydenlyh.github.io/InPO-project-page/)

![photo](./assets/smpo.png "SmPO")

The repository provides the official implementation, experiment code, and model checkpoints used in our research paper.

---

## 📖 News & Updates
- **[2025-06-03]** 🎉 Preprint paper released on arXiv!
- **[2025-06-03]** ✅ Initial model checkpoints published
- **[2025-06-04]** 📊 Project page
- **[2025-06-29]** 🚀 Training code release 

---

## 🔧 Quick Start

### Installation 
```bash
conda create -n smpo python=3.10
conda activate smpo
git clone https://github.com/JaydenLyh/SmPO.git
cd SmPO
pip install -r requirements.txt
```
---
### Preparation of dataset and base models
```bash
SmPO/
├── assets/                   
│   └── smpo.png            
├── checkpoints/    
│   ├── CLIP-ViT-H-14-laion2B-s32B-b79K/  
│   ├── PickScore_v1/          
│   ├── stable-diffusion-v1-5/          
│   ├── sdxl-vae-fp16-fix/            
│   └── stable-diffusion-xl-base-1.0/         
├── datasets/                 
│   └── pickapic_v2/   
├── utils/  
│   └── pickscore_utils.py  
├── train.py            
├── README.md              
├── LICENSE.txt            
└── requirements.txt       
```
---
### Step 1: Smooth Pick-a-Pic v2
```bash
python preprocessing.py
```
---
### Step 2: Training for SDXL
```bash
export MODEL_NAME="checkpoints/stable-diffusion-xl-base-1.0"
export VAE="checkpoints/sdxl-vae-fp16-fix"
export DATASET_NAME="pickapic_v2"
PORT=$((20000 + RANDOM % 10000))

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --main_process_port $PORT --mixed_precision="fp16" --num_processes=8 train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=200 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 50 \
  --beta_dpo 5000 \
  --sdxl  \
  --output_dir="smpo-sdxl" 
```
---

## Our Models
| Model          | Download Links                          
|----------------|-----------------------------------------|
| SmPO-SD1.5     | [Hugging Face](https://huggingface.co/JaydenLu666/SmPO-SD1.5)  |
| SmPO-SDXL    |  [Hugging Face](https://huggingface.co/JaydenLu666/SmPO-SDXL)       |

## Citation
```bash
@article{lu2025smoothed,
  title={Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences},
  author={Lu, Yunhong and Wang, Qichao and Cao, Hengyuan and Xu, Xiaoyin and Zhang, Min},
  journal={arXiv preprint arXiv:2506.02698},
  year={2025}
}
```

## Acknowledgments
The implementation of this project references the [DiffusionDPO](https://github.com/SalesforceAIResearch/DiffusionDPO) repository by Salesforce AI Research. We acknowledge and appreciate their open-source contribution.