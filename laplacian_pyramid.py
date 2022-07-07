import cv2
import numpy as np 

def check_image_size(img, lv):
    size = 2**lv
    rows, cols = img.shape[:2]
    assert rows > size and cols > size, 'lv 값을 낮추세요.'

    _rows = rows//size * size
    _cols = cols//size * size
    return _rows, _cols

def crop_image(img, lv):
    rows, cols = check_image_size(img, lv)
    return img[:rows, :cols, ...]

def get_pyramid_images(img, lv):
    img = crop_image(img, lv)
    pyramid_images = [img]
    for i in range(1, lv):
        pyramid_images.append(cv2.pyrDown(pyramid_images[i-1]))
    return pyramid_images

def get_laplacian_pyramid_images(img, lv):
    # lv 마다 1/2 크기로 줄어 들기 때문에 원본 영상 크기로 복원이 안됨. 영상의 크기를 미리 조절하여 사용함
    # float32로 사용할 경우 복원 시 원본 영상과 다를 수 있음 (float64 변경 시 일치함)
    img = crop_image(img, lv).astype('float32')
    pyramid_images = get_pyramid_images(img, lv)
    # 배열의 순서를 변경 
    pyramid_images = pyramid_images[::-1]
    laplacian_pyramid_images = [pyramid_images[0]]
    for i in range(1, len(pyramid_images)):
        laplacian_pyramid_images.append(pyramid_images[i] - cv2.pyrUp(pyramid_images[i - 1]))
    return laplacian_pyramid_images[::-1]

def reconstruct_laplacian_pyramid_images(laplacian_pyramid_images):
    laplacian_pyramid_images = laplacian_pyramid_images[::-1]
    recon = laplacian_pyramid_images[0]
    for i in range(1, len(laplacian_pyramid_images)):
        recon = cv2.pyrUp(recon)
        recon += laplacian_pyramid_images[i]
    recon = np.uint8(np.clip(recon,0, 255))
    return recon

def show_pyramid_images(pyramid_images):
    for i, image in enumerate(pyramid_images):
        image = np.clip(image, 0, 255)
        cv2.imshow(f'lv: {i}', image.astype('uint8'))
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__=='__main__':
    img = cv2.imread('./images/image_blending/apple.jpg')
    lv = 4
    img = crop_image(img, lv)
    l = get_laplacian_pyramid_images(img, lv)
    out = reconstruct_laplacian_pyramid_images(l)
    print(np.array_equal(img, out))
    # show images
    show_pyramid_images(l)
    cv2.imshow('out', np.hstack([img, out]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
