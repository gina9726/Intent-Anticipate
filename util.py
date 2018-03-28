import numpy as np
import pdb
import math

def discount_cumsum(x, discount_rate):
    R = [x[-1]]
    for V in x[:-1][::-1]:
        R.append(V + discount_rate*R[-1])
    
    return np.array(R[::-1])
