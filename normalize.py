import numpy as np 

def clip3ch(img, low=0.01, high=0.99):
    # color 3 channels
    out = np.zeros_like(img)
    for ch in range(3):
        out[...,ch] = clip(img[...,ch], low, high)    
    return out 

def clip(img, low=0.01, high=0.99):   
    # 1d array로 sorting함
    sorted_img = np.sort(img, axis=None)
    length = len(sorted_img)
    low_loc = int(length * low)
    high_loc = int(length * high)

    if low_loc < 0: low_loc = 0 
    if high_loc >= length: high_loc = length - 1

    low_value = sorted_img[low_loc]
    high_vlaue = sorted_img[high_loc]

    clipped_img = (img - low_value) / (high_vlaue - low_value)
    # 정규화 이후 0 ~ 1 사이 값만 남김
    clipped_img = np.clip(clipped_img, 0, 1)

    return clipped_img