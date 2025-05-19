import cv2
import numpy as np
from translate import *

def calculate_max_capacity(image_path):
    """Рассчитывает максимальную вместимость изображения в символах"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Изображение не найдено или путь неверный")
    
    height, width = img.shape
    total_pixels = height * width
    max_bytes = total_pixels // 8
    max_chars = max_bytes - 2  # -2 для маркера конца
    
    return max_chars

def encode_lsb_grayscale_safe(input_image_path, output_image_path, secret_message):
    """
    Скрывает сообщение в черно-белом изображении, обрезая его если нужно
    :param input_image_path: путь к исходному изображению
    :param output_image_path: путь для сохранения изображения со скрытым сообщением
    :param secret_message: сообщение для скрытия (будет обрезано если слишком большое)
    :return: tuple (успешность операции, фактически встроенное сообщение)
    """
    # Читаем изображение в grayscale
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Изображение не найдено или путь неверный")
    
    height, width = img.shape
    max_chars = calculate_max_capacity(input_image_path)
    
    # Обрезаем сообщение если оно слишком длинное
    if len(secret_message) > max_chars:
        truncated_message = secret_message[:max_chars]
        print(f"Предупреждение: Сообщение обрезано до {max_chars} символов")
    else:
        truncated_message = secret_message
    
    # Преобразуем сообщение в бинарный формат
    binary_message = ''.join([format(ord(i), '08b') for i in truncated_message])
    binary_message += '1111111111111110'  # Маркер конца
    
    # Создаем копию изображения для модификации
    encoded_img = img.copy()
    message_index = 0
    
    # Проходим по каждому пикселю
    for i in range(height):
        for j in range(width):
            if message_index < len(binary_message):
                # Заменяем младший бит
                encoded_img[i, j] = (encoded_img[i, j] & 0xFE) | int(binary_message[message_index])
                message_index += 1
            else:
                break
        if message_index >= len(binary_message):
            break
    
    # Сохраняем изображение
    cv2.imwrite(output_image_path, encoded_img)
    return True, truncated_message

def decode_lsb_grayscale(encoded_image_path):
    """Извлекает скрытое сообщение из черно-белого изображения"""
    img = cv2.imread(encoded_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Изображение не найдено или путь неверный")
    
    height, width = img.shape
    binary_message = ""
    
    # Проходим по каждому пикселю
    for i in range(height):
        for j in range(width):
            # Извлекаем младший бит
            binary_message += str(img[i, j] & 1)
            
            # Проверяем маркер конца сообщения
            if len(binary_message) >= 16 and binary_message[-16:] == '1111111111111110':
                # Преобразуем бинарную строку в текст
                message = ""
                for k in range(0, len(binary_message)-16, 8):
                    byte = binary_message[k:k+8]
                    message += chr(int(byte, 2))
                return message
    
    return ""

if __name__ == "__main__":
    # Пример использования
    input_image = "./LSB/cat.png"
    output_image = "./LSB/encoded.png"
    dir_sec_msg = "./LSB/input_message.txt"
    
    # Читаем сообщение из файла
    secret_msg = read_text_from_file(dir_sec_msg)
    
    # Сохраняем битовое представление (для отладки)
    save_text_to_file("./LSB/input_message_bit.txt", text_to_bits(secret_msg))
    
    try:
        # Пытаемся встроить сообщение (с автоматическим обрезанием если нужно)
        success, embedded_msg = encode_lsb_grayscale_safe(input_image, output_image, secret_msg)
        
        if success:
            print(f"Успешно встроено {len(embedded_msg)} символов из {len(secret_msg)}")
            
            # Извлекаем сообщение для проверки
            decoded = decode_lsb_grayscale(output_image)
            print(f"Извлечённое сообщение: '{decoded}'")
            
            # Визуализация
            original = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
            encoded = cv2.imread(output_image, cv2.IMREAD_GRAYSCALE)
            
            cv2.imshow('Original', original)
            cv2.imshow('Encoded', encoded)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка: {e}")