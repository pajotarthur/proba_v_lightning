from __future__ import print_function

import numpy as np
import skimage
import torch.optim
from DIP.models.resnet import ResNet
from DIP.utils.inpainting_utils import *
from skimage import io

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
from pathlib import Path

p = Path('/local/pajot/data/proba_v/train/NIR/imgset0594/')
img_path = p / 'LR000.png'
mask_path = p / 'QM000.png'

PLOT = True
imsize = -1
dim_div_by = 64
NET_TYPE = 'skip_depth6'  # one of skip_depth4|skip_depth2|UNET|ResNet

img_pil, img_np = get_image(img_path, imsize)
img_mask_pil, img_mask_np = get_image(mask_path, imsize)

img_mask_pil = crop_image(img_mask_pil, dim_div_by)
img_pil = crop_image(img_pil, dim_div_by)

img_np = pil_to_np(img_pil)
img_mask_np = pil_to_np(img_mask_pil)

img_np = np.array(io.imread(img_path), dtype=np.uint16)[None, ...]
img_mask_np = np.array(io.imread(mask_path), dtype=np.bool)[None, ...]

pad = 'reflection'  # 'zero'
OPT_OVER = 'net'
OPTIMIZER = 'adam'

INPUT = 'meshgrid'
input_depth = 2
LR = 0.001
num_iter = 3000
param_noise = False
reg_noise_std = 0.00

net = ResNet(input_depth, img_np.shape[0], 8, 32, need_sigmoid=True, act_fun='LeakyReLU')

net = net.type(dtype)
net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)

# Compute number of parameters
s = sum(np.prod(list(p.size())) for p in net.parameters())
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_var = torch.from_numpy(skimage.img_as_float(img_np).astype(np.float32)).to('cuda:0')
mask_var = torch.from_numpy(img_mask_np.astype(np.float32)).to('cuda:0')

# hr = torch.from_numpy(skimage.img_as_float(HR_im).astype(np.float32))
# mask_hr = torch.from_numpy(HR_mask.astype(np.float32))
i = 0


def closure():
    global i

    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50

    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    total_loss = mse((out * mask_var).squeeze(), (img_var * mask_var).squeeze())
    total_loss.backward()

    print('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
    i += 1

    return total_loss


net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

pp = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, pp, closure, LR, num_iter)

out_np = torch_to_np(net(net_input))
np.save(p / '0000_dip.npy', out_np)
