import cv2
import numpy as np
#from PIL import Image

print("Hello, World!")


image = cv2.cvtColor(cv2.imread('les.jpg'), cv2.COLOR_BGR2GRAY)
rows, height = image.shape
channels = 1


Nc = (rows//8*height//8)*channels #количество встраиваемых бит


def segdiv(image):
    seg = []
    block_size = 8
    rows, height = image.shape
    print(height, rows, channels)
    Nc = (rows//8*height//8)*channels #количество встраиваемых бит
    for c in range(0, channels):
        for x in range(0, rows, block_size):
            for y in range(0, height, block_size):
                end_x = x + block_size
                end_y = y + block_size
                
                # Проверяем, чтобы блок не выходил за границы изображения
                if end_x <= rows and end_y <= height:
                    seg.append(image[x:end_x, y:end_y])
    return seg

#преобраззование в float32 подавал в uint8 выход тоже в uint8
def dct_blocks(seg):
    dct_seg = []
    seg_float32 = [np.array(block, dtype=np.float32) for block in seg]
    dct_seg = [cv2.dct(block/255.0) for block in seg_float32]
    return dct_seg

#расчет суммы двумерного массива с вычетанием DC коэффициента
def sum_2d_array(arr):
    total_sum = 0
    for row in arr:
        for element in row:
            total_sum += abs(element)
    total_sum = total_sum - arr[0,0]
    return total_sum


#s = dct_blocks(segdiv(image))
#A = sum_2d_array(s[0])

P = 40 #Порог различения
PL = 2600.0
PH = 40.0


#def embedding():
#определение пригодности блока для встраивания 0 - не пригоден 1 - пригоден
#в качестве аргумента принимает один элемент матрицы 8х8 float32
def block_suitability(segItem_f32):
    sum_dct = sum_2d_array(segItem_f32)
    sum_dct = sum_dct*255
    if PH < sum_dct < PL:
        suitable = 1
    else:
        suitable = 0
    return suitable

block_suitability(dct_blocks(segdiv(image))[0])









cv2.imshow('container', image)
cv2.waitKey(0)
cv2.destroyAllWindows()