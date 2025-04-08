import cv2
import numpy as np
#from translate import bit_sequence
#from PIL import Image


print("Hello, World!")


channels = 1
def zigzag_transform(matrix):
    """
    Корректное зигзаг-преобразование для блока 8x8.
    Элемент (1,1) будет на 4-й позиции (индекс 3).
    """
    n = 8
    zigzag = []
    
    for s in range(0, 2*n-1):
        if s < n:
            if s % 2 == 0:
                # Движение вниз-влево ↘
                i, j = s, 0
                while i >= 0 and j < n:
                    zigzag.append(matrix[i][j])
                    i -= 1
                    j += 1
            else:
                # Движение вверх-вправо ↗
                i, j = 0, s
                while i < n and j >= 0:
                    zigzag.append(matrix[i][j])
                    i += 1
                    j -= 1
        else:
            if s % 2 == 0:
                # Движение вниз-влево ↘ (для второй половины)
                i, j = n-1, s - n + 1
                while i >= 0 and j < n:
                    zigzag.append(matrix[i][j])
                    i -= 1
                    j += 1
            else:
                # Движение вверх-вправо ↗ (для второй половины)
                i, j = s - n + 1, n-1
                while i < n and j >= 0:
                    zigzag.append(matrix[i][j])
                    i += 1
                    j -= 1
    
    return np.array(zigzag)

def inverse_zigzag(vector, n):
    """
    Обратное зигзаг-преобразование (вектор -> матрица)
    :param vector: 1D numpy array
    :param n: размер выходной матрицы (n x n)
    :return: 2D numpy array
    """
    matrix = np.zeros((n, n))
    idx = 0
    
    for i in range(2 * n - 1):
        if i < n:
            if i % 2 == 0:
                row, col = i, 0
                while row >= 0 and col < n and idx < len(vector):
                    matrix[row][col] = vector[idx]
                    row -= 1
                    col += 1
                    idx += 1
            else:
                row, col = 0, i
                while row < n and col >= 0 and idx < len(vector):
                    matrix[row][col] = vector[idx]
                    row += 1
                    col -= 1
                    idx += 1
        else:
            if i % 2 == 0:
                row, col = n - 1, i - n + 1
                while row >= 0 and col < n and idx < len(vector):
                    matrix[row][col] = vector[idx]
                    row -= 1
                    col += 1
                    idx += 1
            else:
                row, col = i - n + 1, n - 1
                while row < n and col >= 0 and idx < len(vector):
                    matrix[row][col] = vector[idx]
                    row += 1
                    col -= 1
                    idx += 1
    return matrix

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
                # Проверяем, чтобы блок не выходил за границы изображения
                if end_x <= rows and end_y <= height:
                    block = idct_seg[index]
                    block = np.clip(block*255, 0, 255)  # Масштабируем и обрезаем значения
                    block = block.astype(np.uint8)  # Преобразуем в uint8
                    reconstructed_image[x:end_x, y:end_y] = block
                    index += 1
    return reconstructed_image


#Расчет суммы LF

#Расчет суммы HF

#расчет суммы двумерного массива с вычетанием DC коэффициента
def sum_2d_array(arr):
    total_sum = 0
    for row in arr:
        for element in row:
            total_sum += abs(element)
    total_sum = total_sum - arr[0,0]
    return total_sum



P = 40.0 #Порог различения
PL = 5000.0 #Порог сверху
PH = 5.0 #Порог снизу

u1 = 6
v1 = 2
u2 = 4
v2 = 4
u3 = 2
v3 = 6

#определение пригодности блока для встраивания 0 - не пригоден 1 - пригоден
#в качестве аргумента принимает один элемент матрицы 8х8 float32
def block_suitability(segItem_f32):
    sum_dctLF = sum(abs(zigzag_transform(segItem_f32)[1:27]))*255.0
    sum_dctHF = sum(abs(zigzag_transform(segItem_f32)[43:63]))*255.0
    if sum_dctLF < PL and sum_dctHF > PH:
        suitable = 1
    else:
        suitable = 1#0
    return suitable
  

def print_array(array):
    print("Значение С1: ", array[u1,v1], " Значение С2: ", array[u2,v2]," Значение С3: ", array[u3,v3])
    print("\n----------------------------")


P_dif = (P/2)/255
def embed_bits(dct_seg, bit_sequence):
    i = 0
    embed_image = dct_seg
    for a in embed_image:
        if (block_suitability(a) == 1):
            if (bit_sequence[i] == "0"):
                wmin = min(a[u1,v1],a[u2,v2])
                a[u3,v3] = wmin - P_dif
                if wmin == a[u1,v1]:
                    a[u1,v1] = a[u1,v1] + P_dif
                if wmin == a[u2,v2]:
                    a[u2,v2] = a[u2,v2] + P_dif
                i += 1
            else:
                wmax = max(a[u1,v1],a[u2,v2])
                a[u3,v3] = wmax + P_dif
                if wmax == a[u1,v1]:
                    a[u1,v1] = a[u1,v1] - P_dif
                if wmax == a[u2,v2]:
                    a[u2,v2] = a[u2,v2] - P_dif
                i += 1
    return embed_image

def separ_bits(im):
    cvz_list = []
    segemb_dct = dct_blocks(segdiv(im))
    i = 0
    for a in segemb_dct:
        if (block_suitability(a) == 1):
            a1 =a[u3,v3]
            b = a[u1,v1]
            c = a[u2,v2]
            if a[u3,v3] < min(a[u1,v1], a[u2,v2]):
                cvz_list.append('0')
            elif a[u3,v3] > max(a[u1,v1], (a[u2,v2])):
                cvz_list.append('1')
    return cvz_list

def fill_bits(image_cvz):
    crypted_data = separ_bits(image_cvz)
    with open('./algorithmBMYeYu/output_message_bit.txt', 'w') as file:
        file.write("".join(map(str, crypted_data)))



def encode(image, bit_sequence):
    image_cvz = segpair(idct_blocks(embed_bits(dct_blocks(segdiv(image)), bit_sequence)),image)
    cv2.imwrite('./algorithmBMYeYu/encoded.jpg', image_cvz, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return image_cvz

def decode(image_cvz):
    fill_bits(image_cvz)



