import numpy as np
from PIL import Image
import hashlib
import struct
import os
from tqdm import tqdm

def get_seed(key):
    """Генерация стабильного seed с улучшенным хешированием"""
    return int.from_bytes(hashlib.sha3_256(key.encode()).digest()[:4], 'little')

def embed_message(img_path, message, key, output_path):
    """Улучшенная функция внедрения с двойной проверкой"""
    try:
        # 1. Подготовка сообщения
        msg_bytes = message.encode('utf-8')
        length = len(msg_bytes)
        crc = hashlib.sha3_256(msg_bytes).digest()[:2]
        
        # 2. Упаковка данных
        header = struct.pack('>IH', length, int.from_bytes(crc, 'big'))
        full_data = header + msg_bytes
        bits = np.unpackbits(np.frombuffer(full_data, dtype=np.uint8))
        
        # 3. Загрузка изображения
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        pixels = np.array(img)
        flat = pixels[:, :, 0].copy().flatten()  # Работаем с копией красного канала
        
        # 4. Проверка емкости
        if len(bits) > len(flat):
            raise ValueError(f"Нужно {len(bits)} бит, доступно {len(flat)}")
        
        # 5. Генерация позиций
        rng = np.random.RandomState(get_seed(key))
        positions = np.arange(len(flat))
        rng.shuffle(positions)
        positions = positions[:len(bits)]
        positions.sort()  # Критически важно для совпадения
        
        # 6. Внедрение данных
        for i, pos in enumerate(tqdm(positions, desc="Внедрение")):
            flat[pos] = (flat[pos] & 0xFE) | bits[i]
        
        # 7. Сохранение с проверкой
        pixels[:, :, 0] = flat.reshape(pixels.shape[0], pixels.shape[1])
        img = Image.fromarray(pixels)
        
        # Сохраняем без сжатия
        img.save(output_path, format='PNG', compress_level=0)
        
        # 8. Верификация
        verify_diff = np.sum(pixels[:, :, 0] != np.array(Image.open(output_path))[:, :, 0])
        if verify_diff > 0:
            raise RuntimeError(f"Обнаружено {verify_diff} изменений при верификации")
            
        return True
    
    except Exception as e:
        print(f"Ошибка внедрения: {str(e)}")
        return False

def extract_message(img_path, key):
    """Улучшенная функция извлечения с полной диагностикой"""
    try:
        # 1. Загрузка изображения
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        pixels = np.array(img)
        flat = pixels[:, :, 0].flatten()
        
        # 2. Генерация позиций
        rng = np.random.RandomState(get_seed(key))
        positions = np.arange(len(flat))
        rng.shuffle(positions)
        positions.sort()
        
        # 3. Чтение заголовка (6 байт)
        header_bits = []
        for pos in positions[:48]:
            header_bits.append(str(flat[pos] & 1))
        
        # Преобразуем биты в байты
        header_bytes = bytearray()
        for i in range(0, 48, 8):
            byte = ''.join(header_bits[i:i+8])
            header_bytes.append(int(byte, 2))
        
        # 4. Распаковка заголовка
        length, crc = struct.unpack('>IH', header_bytes[:6])
        
        # 5. Чтение данных
        data_bits = []
        for pos in positions[48:48+length*8]:
            data_bits.append(str(flat[pos] & 1))
        
        data_bytes = bytearray()
        for i in range(0, len(data_bits), 8):
            byte = ''.join(data_bits[i:i+8])
            data_bytes.append(int(byte, 2))
        
        # 6. Проверка CRC
        calc_crc = hashlib.sha3_256(data_bytes).digest()[:2]
        if calc_crc != crc.to_bytes(2, 'big'):
            print(f"Ожидалось CRC: {crc.to_bytes(2, 'big').hex()}")
            print(f"Получено CRC: {calc_crc.hex()}")
            print(f"Первые 10 байт: {data_bytes[:10].hex()}")
            return "Ошибка: контрольная сумма не совпадает"
        
        return data_bytes.decode('utf-8')
    
    except Exception as e:
        return f"Ошибка извлечения: {str(e)}"

# Параметры
input_img = "./DDQM/les.png"
output_img = "./DDQM/les_encoded.png"
message = "Тестовое сообщение для проверки стегосистемы"
key = "секретный_ключ_123!"

# Внедрение сообщения
print("=== ВНЕДРЕНИЕ ===")
if embed_message(input_img, message, key, output_img):
    print("\n=== ИЗВЛЕЧЕНИЕ ===")
    result = extract_message(output_img, key)
    print(f"\nРезультат: {result}")
    print(f"Совпадение: {'✓' if result == message else '×'}")
    print(f"Seed для вашего ключа: {get_seed('секретный_ключ_123!')}")
    with open("./DDQM/les.png", "rb") as f1, open("./DDQM/les_encoded.png", "rb") as f2:
        content1 = f1.read()
        content2 = f2.read()
        print(f"Файлы {'идентичны' if content1 == content2 else 'различаются'}")
    orig = np.array(Image.open("./DDQM/les.png"))
    mod = np.array(Image.open("./DDQM/les_encoded.png"))
    print(f"Изменено пикселей: {np.sum(orig != mod)}")
else:
    print("Не удалось внедрить сообщение")
