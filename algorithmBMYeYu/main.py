import cv2
import numpy as np
#from PIL import Image


#----------------------------------------------------------------------------------------
#для переноса и вызова из другого файла
def text_to_bits(text):
    # Преобразуем каждый символ в битовую последовательность
    bits = ''.join(format(ord(char), '08b') for char in text)
    return bits

def bits_to_text(bits):
    # Разделяем битовую последовательность на группы по 8 бит
    chunks = [bits[i:i+8] for i in range(0, len(bits), 8)]
    
    # Преобразуем каждую группу бит в символ
    text = ''.join(chr(int(chunk, 2)) for chunk in chunks)
    return text

# Загрузка текста из файла
def load_text_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Сохранение текста в файл
def save_text_to_file(filename, text):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

# Пример использования
input_filename = 'lorem_ipsum.txt'  # Имя файла с исходным текстом
output_filename = 'output.txt'  # Имя файла для сохранения результата

# Загружаем текст из файла
text = load_text_from_file(input_filename)
print("Исходный текст:", text)

# Преобразуем текст в битовую последовательность
bit_sequence = text_to_bits(text)
print("Битовая последовательность:", bit_sequence)

# Преобразуем битовую последовательность обратно в текст
restored_text = bits_to_text(bit_sequence)
print("Восстановленный текст:", restored_text)

# Сохраняем восстановленный текст в файл
save_text_to_file(output_filename, restored_text)
print(f"Восстановленный текст сохранен в файл: {output_filename}")

#----------------------------------------------------------------------------------------





print("Hello, World!")


image = cv2.cvtColor(cv2.imread('les.jpg'), cv2.COLOR_BGR2GRAY)
rows, height = image.shape
channels = 1


Nc = (rows//8*height//8)*channels #количество встраиваемых бит


P = 40.0 #Порог различения
PL = 2600.0
PH = 40.0


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

#преобраззование фурье 
#вход в uint8 выход в float32
def dct_blocks(seg):
    dct_seg = []
    seg_float32 = [np.array(block, dtype=np.float32) for block in seg]
    dct_seg = [cv2.dct(block/255.0) for block in seg_float32]
    return dct_seg

#обратное преобразование фурье выход float32 
def idct_blocks(seg):
    idct_seg = []
    #seg_float32 = [np.array(block, dtype=np.float32) for block in seg]
    idct_seg = [cv2.idct(block) for block in seg]
    #idct_normseg = [np.clip(block, 0, 255) for block in idct_seg]
    return idct_seg

def segpair(idct_seg):
    
    block_size = 8
    rows, height = image.shape
    print(height, rows, channels)
    reconstructed_image = image
    Nc = (rows//8*height//8)*channels #количество встраиваемых бит
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


#s = dct_blocks(segdiv(image))  
#A = sum_2d_array(s[0])


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

u1 = 7
v1 = 3
u2 = 5
v2 = 5
u3 = 3
v3 = 7


def iter_bits(bit_sequence):
    if not hasattr(iter_bits, "i"):  # Проверяем, существует ли атрибут
        i = 0  # Инициализируем статическую переменную
    if (bit_sequence[i] == "1"):
        i += 1
        return 1
    else:
        i += 1
        return 0


def embed_bits(dct_seg):
    embed_image = dct_seg
    for a in dct_seg:
        if (block_suitability(a) == 1):
            if (iter_bits == 1):
                wmin = min(a[u1,v1],a[u1,v1])
                a[u3,v3] = wmin - P/2
                if wmin == a[u1,v1]:
                    a[u1,v1] = a[u1,v1] + P/2
                if wmin == a[u2,v2]:
                    a[u2,v2] = a[u2,v2] + P/2
            else:
                wmax = max(a[u1,v1],a[u1,v1])
                a[u3,v3] = wmax + P/2
                if wmax == a[u1,v1]:
                    a[u1,v1] = a[u1,v1] - P/2
                if wmax == a[u2,v2]:
                    a[u2,v2] = a[u2,v2] - P/2
    return embed_image




#iter_bits(bit_sequence)

#block_suitability(dct_blocks(segdiv(image))[0])
#image = segpair(idct_blocks(embed_bits(dct_blocks(segdiv(image)))))
image = segpair(idct_blocks(dct_blocks(segdiv(image))))

cv2.imshow('container', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
