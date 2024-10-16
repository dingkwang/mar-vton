import os

# !git clone https://github.com/LTH14/mar.git
# import os
# os.chdir('mar')
os.environ['PYTHONPATH'] = '/opt/disk2/dwang/vton2/mar'
# !pip install timm==0.9.12
# MAR imports:
import torch
import numpy as np
from models import mar
from models.vae import AutoencoderKL
from torchvision.utils import save_image
from util import download
from PIL import Image
from IPython.display import display

torch.set_grad_enabled(False)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")


model_type = "mar_large"  #@param ["mar_base", "mar_large", "mar_huge"]
num_sampling_steps_diffloss = 100  #@param {type:"slider", min:1, max:1000, step:1}
if model_type == "mar_base":
    # download.download_pretrained_marb(overwrite=False)
    diffloss_d = 6
    diffloss_w = 1024
elif model_type == "mar_large":
    download_pretrained_marl = download.download_pretrained_marl(overwrite=False) 
    diffloss_d = 8
    diffloss_w = 1280
elif model_type == "mar_huge":
    # download.download_pretrained_marh(overwrite=False)
    diffloss_d = 12
    diffloss_w = 1536
else:
    raise NotImplementedError
model = mar.__dict__[model_type](buffer_size=64,
                                 diffloss_d=diffloss_d,
                                 diffloss_w=diffloss_w,
                                 num_sampling_steps=str(num_sampling_steps_diffloss)).to(device)
state_dict = torch.load(download_pretrained_marl)["model_ema"]
model.load_state_dict(state_dict)
model.eval()  # important!
vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path="/opt/disk2/dwang/vton2/mar/pretrained_models/vae/kl16.ckpt").cuda().eval()



# Set user inputs:
seed = 0  #@param {type:"number"}
torch.manual_seed(seed)
np.random.seed(seed)
num_ar_steps = 64  #@param {type:"slider", min:1, max:256, step:1}
cfg_scale = 1  #@param {type:"slider", min:1, max:10, step:0.1}
cfg_schedule = "constant"  #@param ["linear", "constant"]
temperature = 1.0  #@param {type:"slider", min:0.9, max:1.1, step:0.01}
class_labels = 610,  610,  610,  610,610,  610,  610,  610, 834, 834, 834,834,834, 834, 834,834, 445, 445, 445,445,  445, 445, 445,445, 459, 459, 459, 459, 459, 459, 459, 459  #@param {type:"raw"}
samples_per_row = 4  #@param {type:"number"}

with torch.amp.autocast('cuda'):
    sampled_tokens = model.sample_tokens(bsz=len(class_labels),
                                         num_iter=num_ar_steps,
                                         cfg=cfg_scale,
                                         cfg_schedule=cfg_schedule,
                                         labels=torch.Tensor(class_labels).long().cuda(),
                                         temperature=temperature,
                                         progress=True)
    sampled_images = vae.decode(sampled_tokens / 0.2325) # bs, 16, 16, 16 range -5.4 ~ 5.3s

# Save and display images:
save_image(sampled_images, "sample.png", nrow=int(samples_per_row), normalize=True, value_range=(-1, 1))
samples = Image.open("sample.png")
display(samples)
