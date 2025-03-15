from PIL import Image
import numpy as np

#----------------------------------------------------------------------------------------
#текстовое преобразование

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
#преобразование изображения

def image_to_bits(image_path):
    # Открываем изображение
    img = Image.open(image_path)
    # Преобразуем изображение в массив numpy (без параметра copy=False)
    img_array = np.array(img)
    # Преобразуем массив в битовую последовательность
    bits = ''.join(format(pixel, '08b') for row in img_array for pixel in row.flatten())
    return bits, img.size, img.mode

def bits_to_image(bits, size, mode):
    # Преобразуем битовую последовательность обратно в байты
    byte_array = np.array([int(bits[i:i+8], 2) for i in range(0, len(bits), 8)])
    # Восстанавливаем форму массива
    img_array = byte_array.reshape(size[1], size[0], -1)  # height, width, channels
    # Создаем изображение из массива
    img = Image.fromarray(img_array.astype('uint8'), mode)
    return img

# Пример использования
input_image_path = 'logo.jpg'  # Путь к исходному изображению
output_image_path = 'output_image.jpg'  # Путь для сохранения восстановленного изображения

# Преобразуем изображение в битовую последовательность
bits, size, mode = image_to_bits(input_image_path)
print(f"Битовая последовательность (первые 100 бит): {bits[:100]}...")

# Преобразуем битовую последовательность обратно в изображение
restored_image = bits_to_image(bits, size, mode)

# Сохраняем восстановленное изображение
restored_image.save(output_image_path)
print(f"Восстановленное изображение сохранено в: {output_image_path}")