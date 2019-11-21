import itertools
import numpy as np

# For GAN computation
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# Formula described in the challenge description
def cPSNR(sr, hr, mask):
    b = ((hr - sr) / mask.sum()).sum()
    cMSE = np.square(hr - (sr + b)).sum() / mask.sum()
    cPSNR = -10 * np.log10(cMSE)
    return cPSNR


def max_cPSNR(sr, hr, mask):
    size = 378
    sr = sr[3:size + 3, 3:size + 3]
    max_psnr = 0
    for i, j in itertools.product(range(6), range(6)):
        hr_small = hr[i:i + size, j:j + size]
        mask_small = mask[i:i + size, j:j + size]
        psnr = cPSNR(sr, hr_small, mask_small)
        if psnr > max_psnr:
            max_psnr = psnr
    return max_psnr
