import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

def sinewave(rows, cols, T):
    # T: 주기
    y = np.arange(rows)
    y = np.sin(y / T * 2 * np.pi).reshape(rows,1)
    x = np.ones(shape = (1, cols))
    return np.matmul(y,x)

def squarewave(rows, cols, T):
     # T: 주기 (짝수만)
    assert T % 2 == 0, 'T는 짝수로 입력하세요'
    y = np.arange(rows) // (T / 2)
    y = np.where(y % 2 == 0, 1, 0).reshape(rows,1)
    x = np.ones(shape = (1, cols))
    return np.matmul(y,x)

def rectangle(rows, cols, rect_height, rect_width):
    img = np.zeros((rows, cols))
    y0 = rows//2 - rect_height//2
    y1 = y0 + rect_height
    x0 = cols//2 - rect_width//2
    x1 = x0 + rect_width
    img[y0:y1, x0:x1] = 1
    return img

def rotate(img, angle):
    rows, cols = img.shape[:2]
    m = cv2.getRotationMatrix2D(center=(cols//2, rows//2), angle=angle, scale=1.0)
    return cv2.warpAffine(img, m, (cols, rows))

if __name__=='__main__':
    rows, cols = 100,100

    img = rotate(rectangle(rows,cols,10,20), 30)
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    mag = np.abs(fft_shift)
    ang = np.angle(fft_shift)

    figure, axis = plt.subplots(1,3)
    axis[0].imshow(img, cmap='gray')
    axis[0].set_title('input')
    axis[1].imshow(mag, cmap='gray')
    axis[1].set_title('magnitude')
    axis[2].imshow(ang, cmap='gray')
    axis[2].set_title('angle')

    figure, axis = plt.subplots(1,3)
    axis[0].imshow(mag, cmap='gray')
    axis[0].set_title('magnitude')
    axis[1].plot(range(rows), mag[:, cols//2])
    axis[1].set_title('vertical line')
    axis[2].plot(range(cols), mag[rows//2,:])
    axis[2].set_title('horizontal line')
    plt.show()
