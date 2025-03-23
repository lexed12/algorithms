import cv2
import numpy as np
#from translate import bit_sequence
#from PIL import Image


print("Hello, World!")


channels = 1

def segdiv(image):
    seg = []
    block_size = 8
    rows, height = image.shape
    for c in range(0, channels):
        for x in range(0, rows, block_size):
            for y in range(0, height, block_size):
                end_x = x + block_size
                end_y = y + block_size
                
                # Проверяем, чтобы блок не выходил за границы изображения
                if end_x <= rows and end_y <= height:
                    seg.append(image[x:end_x, y:end_y])
    return seg

#преобраззование фурье 
#вход в uint8 выход в float32
def dct_blocks(seg):
    dct_seg = []
    seg_float32 = [np.array(block, dtype=np.float32) for block in seg]
    dct_seg = [cv2.dct(block/255.0) for block in seg_float32]
    return dct_seg

#обратное преобразование фурье выход float32 
def idct_blocks(seg):
    idct_seg = [cv2.idct(block) for block in seg]
    return idct_seg

#склеивание изображения из блоков
def segpair(idct_seg, image):
    block_size = 8
    rows, height = image.shape
    print(height, rows, channels)
    reconstructed_image = image
    index = 0
    for c in range(0, channels):
        for x in range(0, rows, block_size):
            for y in range(0, height, block_size):
                end_x = x + block_size
                end_y = y + block_size
                block = idct_seg[index]
                block = np.clip(block*255, 0, 255)  # Масштабируем и обрезаем значения
                block = block.astype(np.uint8)  # Преобразуем в uint8
                # Проверяем, чтобы блок не выходил за границы изображения
                if end_x <= rows and end_y <= height:
                    reconstructed_image[x:end_x, y:end_y] = block
                    index += 1            
    return reconstructed_image

#расчет суммы двумерного массива с вычетанием DC коэффициента
def sum_2d_array(arr):
    total_sum = 0
    for row in arr:
        for element in row:
            total_sum += abs(element)
    total_sum = total_sum - arr[0,0]
    return total_sum



P = 50.0 #Порог различения
PL = 2600.0
PH = 800.0

u1 = 7
v1 = 3
u2 = 5
v2 = 5
u3 = 3
v3 = 7

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

def iter_bits(bit_sequence):
    if not hasattr(iter_bits, "i"):  # Проверяем, существует ли атрибут
        i = 0  # Инициализируем статическую переменную
    if (bit_sequence[i] == "1"):
        i += 1
        return "1"
    else:
        i += 1
        return "0"

def embed_bits(dct_seg, bit_sequence):
    i = 0
    embed_image = dct_seg
    for a in embed_image:
        if (block_suitability(a) == 1):
            if (bit_sequence[i] == "1"):
                wmin = min(a[u1,v1],a[u1,v1])
                a[u3,v3] = wmin - P/2
                if wmin == a[u1,v1]:
                    a[u1,v1] = a[u1,v1] + P/2
                if wmin == a[u2,v2]:
                    a[u2,v2] = a[u2,v2] + P/2
                i += 1
            else:
                wmax = max(a[u1,v1],a[u1,v1])
                a[u3,v3] = wmax + P/2
                if wmax == a[u1,v1]:
                    a[u1,v1] = a[u1,v1] - P/2
                if wmax == a[u2,v2]:
                    a[u2,v2] = a[u2,v2] - P/2
                i += 1
    return embed_image

def separ_bits(im):
    cvz_list = []
    segemb_dct = dct_blocks(segdiv(im))
    for a in segemb_dct:
        if (block_suitability(a) == 1):
            a1 =a[u3,v3]
            b = a[u1,v1]
            c =a[u2,v2]

            if a[u3,v3] < min(a[u1,v1], a[u2,v2]):
                cvz_list.append('1')
            elif a[u3,v3] > max(a[u1,v1], a[u2,v2]):
                cvz_list.append('0')
    return cvz_list

def fill_bits(image_cvz):
    crypted_data = separ_bits(image_cvz)
    with open('output_message_bit.txt', 'w') as file:
        file.write("".join(map(str, crypted_data)))



def encode(image, bit_sequence):
    image_cvz = segpair(idct_blocks(embed_bits(dct_blocks(segdiv(image)), bit_sequence)),image)
    return image_cvz

def decode(image_cvz):
    fill_bits(image_cvz)



