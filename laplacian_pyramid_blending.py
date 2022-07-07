import cv2 
import numpy as np 
from laplacian_pyramid import get_laplacian_pyramid_images, get_pyramid_images, reconstruct_laplacian_pyramid_images

def blending(img_left, img_right, mask, lv):
    # mask의 최댓값은 1
    img1_laplacian_pyramid = get_laplacian_pyramid_images(img_left, lv)
    img2_laplacian_pyramid = get_laplacian_pyramid_images(img_right, lv)
    mask_pyramid = get_pyramid_images(mask,lv)

    blending = []
    for i in range(lv):
        # mask의 1은 img1의 값을 가져오고 maskd의 0은 img2의 값을 가져옴    
        p = mask_pyramid[i] * img1_laplacian_pyramid[i] + (1-mask_pyramid[i]) * img2_laplacian_pyramid[i]
        blending.append(p)
    out = reconstruct_laplacian_pyramid_images(blending)
    return out 

if __name__=='__main__':
    apple = cv2.imread('./images/image_blending/apple.jpg')
    orange = cv2.imread('./images/image_blending/orange.jpg')
    # 두 영상을 좌우로 블랜딩하기 위한 마스크 형태 
    mask = np.zeros_like(apple)
    rows, cols = mask.shape[:2]
    mask[:,:cols//2] = 1
    pyramid_blending = blending(apple, orange, mask, 5)
    # 비교용
    rows, cols = apple.shape[:2]
    simple = np.hstack([apple[:,:cols//2,:], orange[:,cols//2:,:]])
    cv2.imshow('input',np.hstack([apple, orange]))
    cv2.imshow('pyramid blending',pyramid_blending)
    cv2.imshow('simple',simple)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


