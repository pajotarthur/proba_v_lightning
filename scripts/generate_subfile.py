import os
from pathlib import Path
from zipfile import ZipFile

import torch
from skimage import io, img_as_uint
from tqdm import tqdm

from data import ProbaDataset
from models.SRResNet import MSRResNet

data_directory = Path('.../proba_v/test')
test_dataset = ProbaDataset(root=data_directory, train_test_val='test', top_k=11, rand=False, stat=True)

checkpoint = torch.load(".../checkpoints/_ckpt_epoch_197.ckpt")
model = MSRResNet(in_nc=13, out_nc=1, nf=48, nb=16, upscale=3)
checkpoint_state = {k.replace('gen.', ''): v for k, v in checkpoint['state_dict'].items() if 'gen' in k}

model.load_state_dict(checkpoint_state)

out = Path('submission')
os.makedirs(out, exist_ok=True)

for lr, name, sr in tqdm(test_dataset):
    lr = lr.unsqueeze(0)
    sr = model(lr, sr)[:, 0]
    sr = sr.detach().cpu().numpy()[0]
    sr = img_as_uint(sr)
    io.imsave(out /name / '.png', sr)
    print('*', end='', flush='True')

sub_archive = out / 'sub.zip'
with ZipFile(sub_archive, mode='w') as zip:
    for img in out.iterdir():
        if not img.suffix == ('.png'):
            zip.write(os.path.join(out, img), arcname=img)
        print('*', end='', flush='True')
