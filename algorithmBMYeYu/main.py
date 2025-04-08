import cv2
import numpy as np
from bmeu import *
from translate import *
import math
import os




image = cv2.cvtColor(cv2.imread('./algorithmBMYeYu/les.jpg'), cv2.COLOR_BGR2GRAY)
rows, height = image.shape

def bits_to_numbers(bits):
    return [int(bit) for bit in bits]

def calculate_mse(original, decoded):
    if len(original) != len(decoded):
        raise ValueError("Длины исходного и декодированного сообщений должны совпадать.")
    
    squared_errors = [(o - d) ** 2 for o, d in zip(original, decoded)]
    mse = sum(squared_errors) / len(original)
    return mse

def read_bits_from_file(filename, max_bits=None):
    """Читает биты из файла и возвращает строку битов."""
    with open(filename, 'r') as file:
        bits = file.read().strip()  # Читаем всё содержимое файла
    if max_bits is not None:
        bits = bits[:max_bits]  # Обрезаем, если указано ограничение
    return bits

def limit_decoded_bits(original_bits, decoded_length):
    """Ограничивает исходное сообщение по длине декодированного."""
    if len(original_bits) > decoded_length:
        return original_bits[:decoded_length]  # Обрезаем лишние биты
    else:
        return original_bits.ljust(decoded_length, '0')  # Дополняем нулями

def calculate_psnr(image1, image2):
    """
    Вычисляет PSNR между двумя изображениями.

    :param image1: Исходное изображение (numpy array).
    :param image2: Закодированное изображение (numpy array).
    :return: Значение PSNR в децибелах (dB).
    """
    # Проверяем, что изображения имеют одинаковый размер
    if image1.shape != image2.shape:
        raise ValueError("Изображения должны иметь одинаковый размер.")

    # Вычисляем MSE
    mse = np.mean((image1 - image2) ** 2)

    # Если MSE равно 0, PSNR бесконечен
    if mse == 0:
        return float('inf')

    # Максимальное значение пикселя (для 8-битных изображений это 255)
    max_pixel = 255.0

    # Вычисляем PSNR
    psnr = 10 * math.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_correct_bits_percentage(original_bits, decoded_bits):
    """
    Вычисляет процент верных бит между двумя битовыми последовательностями.
    
    :param original_bits: Исходная битовая последовательность (строка из 0 и 1).
    :param decoded_bits: Декодированная битовая последовательность (строка из 0 и 1).
    :return: Процент верных бит.
    """
    if len(original_bits) != len(decoded_bits):
        raise ValueError("Длины исходной и декодированной последовательностей должны совпадать.")
    
    # Считаем количество совпавших бит
    correct_bits = sum(o == d for o, d in zip(original_bits, decoded_bits))
    print("Количество правильных бит: " ,correct_bits)
    # Вычисляем процент верных бит
    total_bits = len(original_bits)
    percentage = (correct_bits / total_bits) * 100
    
    return percentage

save_text_to_file('./algorithmBMYeYu/input_message_bit.txt',text_to_bits(read_text_from_file('./algorithmBMYeYu/input_message.txt')))
bit_sequence = read_text_from_file('./algorithmBMYeYu/input_message_bit.txt')


container = encode(image, bit_sequence)
decode(container)


# Чтение исходного сообщения из файла
original_bits = read_bits_from_file('./algorithmBMYeYu/input_message_bit.txt')

# Чтение декодированного сообщения из файла
decoded_bits = read_bits_from_file('./algorithmBMYeYu/output_message_bit.txt', max_bits=len(original_bits))
save_text_to_file("./algorithmBMYeYu/output_message.txt",bits_to_text(decoded_bits))

# Ограничиваем декодированное сообщение по длине исходного
original_bits = limit_decoded_bits(original_bits, len(decoded_bits))

# Преобразуем биты в числа
original_numbers = bits_to_numbers(original_bits)
decoded_numbers = bits_to_numbers(decoded_bits)

# Вычисляем MSE
mse = calculate_mse(original_numbers, decoded_numbers)
print("MSE:", mse)

# Пример использования
# Загружаем изображения
original_image = cv2.imread('./algorithmBMYeYu/les.jpg', cv2.IMREAD_GRAYSCALE)
encoded_image = cv2.imread('./algorithmBMYeYu/encoded.jpg', cv2.IMREAD_GRAYSCALE)

# Вычисляыем PSNR
psnr_value = calculate_psnr(original_image, container)
print(f"PSNR: {psnr_value:.2f} dB")

percentage = calculate_correct_bits_percentage(original_numbers, decoded_numbers)
print(f"Процент верных бит: {percentage:.2f}%")

