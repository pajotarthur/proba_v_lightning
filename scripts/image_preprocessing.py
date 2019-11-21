import pathlib
from itertools import chain

import numpy as np
from skimage import io
from tqdm import tqdm

# %%

p = pathlib.Path("/local/pajot/data/proba_v/")
p_train = p / 'train'
p_test = p / 'test'

p_train_red = p_train / 'RED'
p_train_nir = p_train / 'NIR'
p_test_red = p_test / 'RED'
p_test_nir = p_test / 'NIR'

l = chain(p_test_nir.iterdir(), p_test_nir.iterdir(), p_train_nir.iterdir(), p_train_red.iterdir())

for imgset in tqdm(l):
    list_SR = sorted(imgset.glob('LR*'))
    list_QM = sorted(imgset.glob('QM*'))

    arr_SR = np.zeros([35, 128, 128])
    arr_QM = np.zeros([35, 128, 128])

    for i, n in enumerate(list_SR):
        arr_SR[i] = np.array([io.imread(n)], dtype=np.uint16)
    for i, n in enumerate(list_QM):
        arr_QM[i] = np.array([io.imread(n)], dtype=np.uint16)

    np.save(imgset / 'arr_SR', arr_SR)
    np.save(imgset / 'arr_QM', arr_QM)
