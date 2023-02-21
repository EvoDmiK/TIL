from imutils.paths import list_files
import pandas as pd
import numpy as np
import random
import time
import os

dumm = np.zeros([200, 200], dtype = np.uint8)
print(dumm)

random.seed(time.ctime())

cnt = 0
while np.count_nonzero(dumm == 0) != 0:
    
    random_row = random.randint(0, dumm.shape[0] - 1)
    random_col = random.randint(0, dumm.shape[1] - 1)

    dumm[random_row][random_col] += 1

    cnt += 1

    print(f'[step {cnt}]\n==========\n{dumm}\n\n')
    print(np.count_nonzero(dumm == 0), '\n')

print(f'total count : {cnt}')