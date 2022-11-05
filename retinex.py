import cv2
import numpy as np 
from normalize import clip3ch,clip

def ssr(img, sigma):
    # w의 값에 따라 compression level이 변함
    blur = cv2.GaussianBlur(img, (0,0), sigma)    
    w = 10
    return  np.log10(img * w + 1) - np.log10(blur * w + 1)

def msr(img, sigmas=[15,80,250]):

    if img.dtype == 'uint8':
        img = (img/255.0).astype('float32')

    out = 0
    for sigma in sigmas:
        out += ssr(img, sigma)
    return out/len(sigmas)

if __name__ =='__main__':
    img = cv2.imread('./images/desk.jpg')
    rows, cols = img.shape[:2]
    img = cv2.resize(img, (cols//4, rows//4))
    # b,g,r 채널마다 clip
    out3ch = clip3ch(msr(img))
    # bgr 전체 채널에 대하여 clip
    out = clip(msr(img))

    cv2.imshow('input', img)
    cv2.imshow('retinex_clip3ch', out3ch)
    cv2.imshow('retinex', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




