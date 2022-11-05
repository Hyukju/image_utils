import cv2
import numpy as np 
from normalize import clip3ch

def ssr(img, sigma):
    blur = cv2.GaussianBlur(img, (0,0), sigma)    
    return  np.log(img + 1) - np.log(blur + 1)

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
    out = clip3ch(msr(img))

    cv2.imshow('input', img)
    cv2.imshow('retinex', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




