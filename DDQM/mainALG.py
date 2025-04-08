from PIL import Image
import numpy as np
import os

def classify_zones(image, zone_size):
    width, height = image.size
    zones = []
    
    for y in range(0, height, zone_size):
        for x in range(0, width, zone_size):
            box = (x, y, min(x + zone_size, width), min(y + zone_size, height))
            zone = image.crop(box)
            avg_brightness = np.mean(np.array(zone))
            zones.append((avg_brightness, box))
    
    return zones

def embed_message(image_path, message, zone_size):
    img = Image.open(image_path)
    zones = classify_zones(img, zone_size)
    
    # Сортируем зоны по яркости
    zones.sort(key=lambda z: z[0])
    
    # Преобразуем сообщение в двоичный вид
    binary_message = ''.join(format(ord(char), '08b') for char in message) + '11111111'  # Завершающий байт
    message_index = 0
    
    for brightness, box in zones:
        if message_index < len(binary_message):
            # Извлекаем пиксели зоны
            zone = img.crop(box)
            pixels = np.array(zone)
            
            # Изменяем яркость пикселей для встраивания сообщения
            for i in range(pixels.shape[0]):
                for j in range(pixels.shape[1]):
                    if message_index < len(binary_message):
                        # Встраиваем бит сообщения в яркость пикселя
                        if binary_message[message_index] == '1':
                            pixels[i, j] = np.clip(pixels[i, j] + 1, 0, 255)
                        else:
                            pixels[i, j] = np.clip(pixels[i, j] - 1, 0, 255)
                        message_index += 1
            
            # Сохраняем изменённую зону обратно в изображение
            img.paste(Image.fromarray(pixels), box)
    
    output_path = os.path.join(os.path.dirname(image_path), 'output_image.png')
    img.save(output_path)
    print(f"Изображение сохранено как: {output_path}")

def extract_message(image_path, zone_size):
    img = Image.open(image_path)
    zones = classify_zones(img, zone_size)
    
    # Сортируем зоны по яркости
    zones.sort(key=lambda z: z[0])
    
    binary_message = ''
    
    for brightness, box in zones:
        # Извлекаем пиксели зоны
        zone = img.crop(box)
        pixels = np.array(zone)
        
        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                # Извлекаем последний бит яркости пикселя
                last_bit = pixels[i, j] % 2
                binary_message += str(last_bit)
                
                # Проверяем, достигли ли мы конца сообщения
                if len(binary_message) >= 8 and binary_message[-8:] == '11111111':
                    # Завершающий байт найден
                    return ''.join(chr(int(binary_message[i:i+8], 2)) for i in range(0, len(binary_message)-8, 8))
    
    return None

# Параметры
image_path = './DDQM/les.png'  # Путь к входному изображению
message = 'Hello, World!'       # Сообщение для встраивания
zone_size = 10                  # Размер зоны

# Встраивание сообщения
embed_message(image_path, message, zone_size)

# Извлечение сообщения
extracted_message = extract_message('./DDQM/output_image.png', zone_size)
print("Извлеченное сообщение:")
print(extracted_message)
