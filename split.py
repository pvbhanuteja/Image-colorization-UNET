import os
source1 = "/home/grads/b/bhanu/img_color/landscape_images"
dest11 = "/home/grads/b/bhanu/img_color/data/val/all"
dest12 = "/home/grads/b/bhanu/img_color/data/train/all"
files = os.listdir(source1)
import shutil
import numpy as np
for f in files:
    if np.random.rand(1) < 0.1:
        shutil.copy(source1 + '/'+ f, dest11 + '/'+ f)
    else:
        shutil.copy(source1 + '/'+ f, dest12 + '/'+ f)