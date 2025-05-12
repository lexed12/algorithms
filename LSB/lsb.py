import cv2
import numpy as np
from translate import *

def check_message_capacity(image_path, message):
    """Проверяет, поместится ли сообщение в изображение и выводит информацию"""
    # Читаем изображение в grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Изображение не найдено или путь неверный")
    
    height, width = img.shape
    total_pixels = height * width
    
    # Вычисляем доступную вместимость
    max_bits = total_pixels
    max_bytes = max_bits // 8
    max_chars = max_bytes - 2  # -2 для маркера конца
    
    # Длина сообщения в битах (каждый символ = 8 бит)
    required_bits = len(message) * 8 + 16  # +16 для маркера конца
    
    print("\n=== Анализ вместимости ===")
    print(f"Размер изображения: {width}x{height} пикселей")
    print(f"Всего пикселей: {total_pixels:,}")
    print(f"Макс. вместимость: {max_chars:,} символов")
    print(f"Длина вашего сообщения: {len(message)} символов")
    print(f"Требуется бит: {required_bits} из доступных {max_bits}")
    
    if required_bits > max_bits:
        excess = required_bits - max_bits
        print(f"\n⚠️ Ошибка: Сообщение слишком большое!")
        print(f"Превышение на {excess} бит (~{excess//8} символов)")
        return False
    else:
        remaining = max_bits - required_bits
        print(f"\n✓ Сообщение поместится!")
        print(f"Останется свободных бит: {remaining} (~{remaining//8} символов)")
        return True

def encode_lsb_grayscale(input_image_path, output_image_path, secret_message):
    """
    Скрывает сообщение в черно-белом изображении используя LSB-метод
    :param input_image_path: путь к исходному изображению (например, 'cat.png')
    :param output_image_path: путь для сохранения изображения со скрытым сообщением
    :param secret_message: сообщение для скрытия
    """
    # Читаем изображение в grayscale
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Изображение не найдено или путь неверный")
    
    height, width = img.shape
    
    # Преобразуем сообщение в бинарный формат
    binary_message = ''.join([format(ord(i), '08b') for i in secret_message])
    message_length = len(binary_message)
    
    # Проверяем, поместится ли сообщение в изображение
    max_message_size = height * width  # Только один канал
    if message_length > max_message_size:
        raise ValueError(f"Сообщение слишком большое. Максимум: {max_message_size//8} символов")
    
    # Добавляем маркер конца сообщения
    binary_message += '1111111111111110'  # 0xFFFE в бинарном виде
    
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
    print(f"Сообщение успешно скрыто в {output_image_path}")

def decode_lsb_grayscale(encoded_image_path):
    """
    Извлекает скрытое сообщение из черно-белого изображения
    :param encoded_image_path: путь к изображению со скрытым сообщением
    :return: извлеченное сообщение
    """
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

def encode_lsb_grayscale_checked(input_image_path, output_image_path, secret_message):
    """Скрывает сообщение с предварительной проверкой"""
    if not check_message_capacity(input_image_path, secret_message):
        return False
    
    try:
        encode_lsb_grayscale(input_image_path, output_image_path, secret_message)
        return True
    except Exception as e:
        print(f"Ошибка при кодировании: {e}")
        return False

if __name__ == "__main__":
    # Пример использования
    input_image = "./LSB/cat.png"  # Черно-белая версия изображения
    output_image = "./LSB/encoded.png"
    dir_sec_msg = "./LSB/input_message.txt"
    secret_msg = "Secret_message"
    

    save_text_to_file("./LSB/input_message_bit.txt", text_to_bits(read_text_from_file(dir_sec_msg)))
    secret_msg = read_text_from_file("./LSB/input_message.txt")

    try:
        if encode_lsb_grayscale_checked(input_image, output_image, secret_msg):
            decoded = decode_lsb_grayscale(output_image)
            #print(f"\nУспешно! Извлечённое сообщение: '{decoded}'")
            
            # Визуализация
            original = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
            encoded = cv2.imread(output_image, cv2.IMREAD_GRAYSCALE)
            
            cv2.imshow('Original', original)
            cv2.imshow('Encoded', encoded)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Ошибка: {e}")