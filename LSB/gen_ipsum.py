import lorem

def generate_huge_lorem_to_file(file_path, size_bytes):
    """Генерирует огромный Lorem Ipsum в одну строку и пишет в файл по частям."""
    chunk_size = 1024 * 1024  # 1 МБ за итерацию (для оптимизации)
    with open(file_path, 'w', encoding='utf-8') as f:
        written_bytes = 0
        while written_bytes < size_bytes:
            # Генерируем кусок текста без переносов
            chunk = lorem.paragraph().replace('\n', ' ')
            chunk_bytes = len(chunk.encode('utf-8'))
            
            # Записываем, если не превысим лимит
            if written_bytes + chunk_bytes <= size_bytes:
                f.write(chunk)
                written_bytes += chunk_bytes
            else:
                # Обрезаем последний кусок
                remaining_bytes = size_bytes - written_bytes
                truncated_chunk = chunk.encode('utf-8')[:remaining_bytes].decode('utf-8', errors='ignore')
                f.write(truncated_chunk)
                break

# Пример: создаем файл 10 МБ (10 * 1024 * 1024 байт)
generate_huge_lorem_to_file("./LSB/input_message.txt", 1 * 1024 * 1024)