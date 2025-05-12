import cv2
import numpy as np
#from PIL import Image
from typing import Dict, List, Tuple

print("Hello, World!")

image = cv2.cvtColor(cv2.imread('./algorithmBMYeYu/cat.png'), cv2.COLOR_BGR2GRAY)
rows, height = image.shape
channels = 1
#--------------------------------------------------------------------------
#Разделение изображения
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

#--------------------------------------------------------------------------
#Классификация блоков

high_thresh=50
low_thresh=10

def compute_gx(block):
    """Вычисляет горизонтальный градиент (аналог Sobel Gx)."""
    gx = np.zeros_like(block, dtype=np.float32)
    gx[:, 1:-1] = block[:, 2:] - block[:, :-2]  # Центральные пиксели
    # Края (можно проигнорировать или обработать отдельно)
    return gx

def compute_gy(block):
    """Вычисляет вертикальный градиент (аналог Sobel Gy)."""
    gy = np.zeros_like(block, dtype=np.float32)
    gy[1:-1, :] = block[2:, :] - block[:-2, :]  # Центральные пиксели
    # Края (можно проигнорировать или обработать отдельно)
    return gy

def compute_gradient_magnitude(gx, gy):
    """Вычисляет модуль градиента."""
    return np.sqrt(gx ** 2 + gy ** 2)

def classify_block(block):
    """
    Классифицирует блок 8×8 на три категории:
    - "sharp" (резковыраженный контраст),
    - "smooth" (постепенный контраст),
    - "noisy" (шумовой контраст).
    """
    gx = compute_gx(block)
    gy = compute_gy(block)
    grad_mag = compute_gradient_magnitude(gx, gy)
    
    max_grad = np.max(grad_mag)
    mean_grad = np.mean(grad_mag)
    
    if max_grad > high_thresh:
        sharpness = 'sharp'
        return {sharpness ,mean_grad}   #Резкий контраст
    elif mean_grad < low_thresh:
        sharpness = 'noise'
        return {'noise',mean_grad}    #Шумовой контраст
    else:
        sharpness = 'smooth'
        return {'smooth',mean_grad}    #Плавный контраст

def list_classify_block(image):
    seg_class = []
    seg = segdiv(image)
    for i in seg:
        seg_class.append(classify_block(i))
    return seg_class

def pixel_classify(block):
    zone_block = np.zeros((8, 8), dtype=int)  # Создаём пустой блок 8x8
    const_B = 0.01
    mean_grad, sharp = classify_block(block)  # Предполагается, что эта функция определена где-то ещё
    
    for i in range(8):  # Проходим по строкам
        for j in range(8):  # Проходим по столбцам
            if sharp == 'sharp' or sharp == 'smooth':  
                if block[i, j] <= mean_grad - const_B:
                    zone_block[i, j] = 1
                elif block[i, j] >= mean_grad + const_B:
                    zone_block[i, j] = 2
                else:
                    zone_block[i, j] = 3
            elif sharp == 'noise':
                if block[i, j] <= mean_grad:
                    zone_block[i, j] = 1
                else:
                    zone_block[i, j] = 2
                    
    return zone_block

#--------------------------------------------------------------------------
#Создание масок

def create_checker_mask(block_size: int, subblock_size: int) -> np.ndarray:
    """
    Создает шахматную маску для блока
    :param block_size: размер основного блока (например, 8)
    :param subblock_size: размер подблока (например, 4)
    :return: маска с чередующимися областями 0 и 1
    """
    if block_size % subblock_size != 0:
        raise ValueError("block_size должен быть кратен subblock_size")
    
    mask = np.zeros((block_size, block_size), dtype=np.uint8)
    num_subblocks = block_size // subblock_size
    
    for i in range(num_subblocks):
        for j in range(num_subblocks):
            # Шахматный порядок
            region = (i + j) % 2
            y_start, y_end = i*subblock_size, (i+1)*subblock_size
            x_start, x_end = j*subblock_size, (j+1)*subblock_size
            mask[y_start:y_end, x_start:x_end] = region
    
    return mask

def create_random_mask(block_size: int, subblock_size: int, seed: int = None) -> np.ndarray:
    """
    Создает псевдослучайную маску для блока
    :param block_size: размер основного блока
    :param subblock_size: размер подблока
    :param seed: seed для генератора случайных чисел
    :return: маска с областями 0 и 1
    """
    if block_size % subblock_size != 0:
        raise ValueError("block_size должен быть кратен subblock_size")
    
    if seed is not None:
        np.random.seed(seed)
    
    mask = np.zeros((block_size, block_size), dtype=np.uint8)
    subblocks_per_side = block_size // subblock_size
    total_subblocks = subblocks_per_side ** 2
    
    # Случайное распределение подблоков между областями 0 и 1
    assignments = np.random.randint(0, 2, total_subblocks)
    
    for idx in range(total_subblocks):
        i, j = divmod(idx, subblocks_per_side)
        y_start, y_end = i*subblock_size, (i+1)*subblock_size
        x_start, x_end = j*subblock_size, (j+1)*subblock_size
        mask[y_start:y_end, x_start:x_end] = assignments[idx]
    
    return mask





#--------------------------------------------------------------------------
#Суммирование
def sum_regions(block, mask):
    sum0 = np.sum(block[mask == 0])
    sum1 = np.sum(block[mask == 1])
    return sum0, sum1

def meansum_insert(block_image, zone_mask, category_mask, message):
    """
    Сканирующее складывание: вычисляет сумму и среднее для зон A/Z и категорий 1/2.
    
    Параметры:
    ----------
    image_block : numpy.ndarray (8x8)
        Блок изображения (значения яркости/градиента и т.д.).
    zone_mask : numpy.ndarray (8x8)
        Маска зон: 'A' и 'Z' (другие значения игнорируются).
    category_mask : numpy.ndarray (8x8)
        Маска категорий: 1 и 2 (другие значения игнорируются).
    
    Возвращает:
    -----------
    numpy.ndarray (2x2)
        Массив вида:
        [
            [Сумма зоны A (категория 1), Сумма зоны A (категория 2)],
            [Среднее зоны Z (категория 1), Среднее зоны Z (категория 2)]
        ]
    """
# Инициализация результата
    E = 15.0
    result = np.zeros((2, 2), dtype=float)
    
    # Определяем маски для зон A и Z
    is_zone_A = (zone_mask == 0)
    is_zone_Z = (zone_mask == 1)
    
    # Определяем маски для категорий 1 и 2
    is_cat1 = (category_mask == 1)
    is_cat2 = (category_mask == 2)
    
    # Средние для зоны A по категориям
    a_cat1_pixels = block_image[is_zone_A & is_cat1]
    a_cat2_pixels = block_image[is_zone_A & is_cat2]
    result[0, 0] = np.mean(a_cat1_pixels) if a_cat1_pixels.size > 0 else 0  # A, категория 1
    result[0, 1] = np.mean(a_cat2_pixels) if a_cat2_pixels.size > 0 else 0  # A, категория 2
    
    # Средние для зоны Z по категориям
    z_cat1_pixels = block_image[is_zone_Z & is_cat1]
    z_cat2_pixels = block_image[is_zone_Z & is_cat2]
    result[1, 0] = np.mean(z_cat1_pixels) if z_cat1_pixels.size > 0 else 0  # Z, категория 1
    result[1, 1] = np.mean(z_cat2_pixels) if z_cat2_pixels.size > 0 else 0  # Z, категория 2
    

    nA1 = a_cat1_pixels.size
    nA2 = a_cat2_pixels.size
    nZ1 = z_cat1_pixels.size
    nZ2 = z_cat2_pixels.size

    # Альтернативный вариант с явным указанием осей:
    n = np.zeros((2, 2), dtype=int)
    n[0, 0] = nA1  # A, категория 1
    n[0, 1] = nA2  # A, категория 2
    n[1, 0] = nZ1  # Z, категория 1
    n[1, 1] = nZ2  # Z, категория 2

    L = [0.0, 0.0]
    l1 = [0.0, 0.0]
    x = 0
    for x in range (2):
        L[x] = (result[0][x]*n[0][x]+result[1][x]*n[1][x])/(n[0][x]+n[1][x])
        if message == 1:
            A = np.array([[n[x, 0], n[x, 1]], [1, -1]])  # Матрица коэффициентов
            b = np.array([L[x] * (n[x, 0] + n[x, 1]), E])  # Вектор правой части
            l1[x] = np.linalg.lstsq(A, b, rcond=None)[0]  # Решение системы
        if message == 0:
            A = np.array([[n[x, 0], n[x, 1]], [-1, 1]])  # Матрица коэффициентов
            b = np.array([L[x] * (n[x, 0] + n[x, 1]), E])  # Вектор правой части
            l1[x] = np.linalg.lstsq(A, b, rcond=None)[0]  # Решение системы
    l1t = np.array(l1).T
    result = l1t-L
    
    return result

def modification_pixel(block_image, zone_mask, category_mask, message):
    N = 8
    i = 0
    j = 0
    block = block_image;
    # Определяем маски для зон A и Z
    is_zone_A = (zone_mask == 0)
    insert_result = meansum_insert(block_image, zone_mask, category_mask, message)
    
    # Определяем маски для категорий 1 и 2
    is_cat1 = (category_mask == 1)
    is_cat2 = (category_mask == 2)

    for i in range(N):
        for j in range(N):
            if is_zone_A[i][j] == True and is_cat1[i][j] == True: #A1
                block[i][j] = block[i][j] + insert_result[1][1]
            if is_zone_A[i][j] == False and is_cat1[i][j] == True: #Z1
                block[i][j] = block[i][j] + insert_result[1][2]
            if is_zone_A[i][j] == True and is_cat2[i][j] == True: #A2
                block[i][j] = block[i][j] + insert_result[2][1]
            if is_zone_A[i][j] == False and is_cat2[i][j] == True: #Z2
                block[i][j] = block[i][j] + insert_result[2][2]
    return block

#склеивание изображения из блоков
def segpair(idct_seg, image):
    block_size = 8
    rows, height = image.shape
    print(height, rows, channels)
    reconstructed_image = image
    index = 0
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

def embedding(image):
    secret_message = 1
    block_emb = []
    block_image = segdiv(image)
    mask = create_checker_mask(8,4)
    for block in block_image:
        block_emb.append(modification_pixel(block,mask,pixel_classify(block),secret_message))
    embed = segpair(block_emb, image)
    cv2.imshow("embed",embed)        




#--------------------------------------------------------------------------
#Вызов функции
if __name__ == "__main__":
    embedding(image)
    
    
    




  
    

