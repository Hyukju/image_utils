import cv2
import numpy as np

def srgb2linear(srgb_img):
    srgb_img = img.astype('float32') / 255.0
    d = 0.04045	
    gamma = ((srgb_img + 0.055)/1.055) ** 2.4
    return np.where(srgb_img > d, gamma, srgb_img / 12.92 )

def linear2srgb(linear_img):
    d = 0.0031308
    gamma = 1.055 * (linear_img) ** (1/2.4) - 0.055
    return np.where(linear_img > d, gamma, linear_img *  12.92)

def ev_control(img, ev=0):
    linear_img = srgb2linear(img)
    ev_img = linear_img * (2 ** ev)
    srgb_img = linear2srgb(ev_img)
    out = np.clip(srgb_img * 255, 0, 255)
    return out.astype('uint8')

if __name__=='__main__':
    ev = 2
    img = cv2.imread('./images/exposure_value/ev0.jpg')
    ev_img =  ev_control(img, ev=ev)
    cv2.imwrite(f'ev_{ev}_cv.png', ev_img)
