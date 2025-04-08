import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime
import numpy as np

def analyze_offset_errors(original, extracted, start_pos=2000):
    """Анализ смещения и характера ошибок после стартовой позиции"""
    if len(original) <= start_pos or len(extracted) <= start_pos:
        print("❌ Последовательности слишком короткие для анализа")
        return
    
    # Выделяем части последовательностей для анализа
    orig_tail = original[start_pos:]
    extr_tail = extracted[start_pos:]
    min_len = min(len(orig_tail), len(extr_tail))
    
    # Тестируем различные смещения
    max_offset_test = 10  # Максимальное смещение для проверки
    offset_results = []
    
    print(f"\n🔍 Анализ смещения для битов после позиции {start_pos}:")
    
    for offset in range(-max_offset_test, max_offset_test + 1):
        if offset < 0:
            # Сравниваем оригинал[сдвиг:] с извлеченным[:сдвиг]
            compare_len = min_len + offset
            orig_part = orig_tail[-offset:]
            extr_part = extr_tail[:compare_len]
        else:
            # Сравниваем оригинал[:сдвиг] с извлеченным[сдвиг:]
            compare_len = min_len - offset
            orig_part = orig_tail[:compare_len]
            extr_part = extr_tail[offset:]
        
        errors = sum(1 for o, e in zip(orig_part, extr_part) if o != e)
        error_percent = (errors / compare_len) * 100 if compare_len > 0 else 100
        offset_results.append((offset, error_percent, compare_len))
        
        print(f"Смещение {offset:3d}: {error_percent:.2f}% ошибок (проверено {compare_len} бит)")
    
    # Находим оптимальное смещение с минимальными ошибками
    best_offset = min(offset_results, key=lambda x: x[1])[0]
    print(f"\n✅ Рекомендуемое смещение: {best_offset} бит")
    
    # Визуализация результатов
    offsets = [x[0] for x in offset_results]
    error_rates = [x[1] for x in offset_results]
    
    plt.figure(figsize=(10, 5))
    plt.plot(offsets, error_rates, 'bo-')
    plt.axvline(x=0, color='r', linestyle='--', label='Без смещения')
    plt.axvline(x=best_offset, color='g', linestyle='-', label='Оптимальное смещение')
    plt.xlabel('Величина смещения (бит)')
    plt.ylabel('Процент ошибок (%)')
    plt.title(f'Зависимость ошибок от смещения (после {start_pos} бит)')
    plt.legend()
    plt.grid(True)
    plt.savefig('offset_analysis.png')
    print("\n📊 График зависимости ошибок от смещения сохранен в offset_analysis.png")
    
    return best_offset

def analyze_error_clusters(positions, window_size=100):
    """Анализ кластеризации ошибок"""
    if not positions:
        return
    
    # Преобразуем позиции ошибок в numpy array
    pos_array = np.array(positions)
    
    # Вычисляем интервалы между ошибками
    intervals = np.diff(pos_array)
    
    print("\n🔎 Анализ распределения ошибок:")
    print(f"Средний интервал между ошибками: {np.mean(intervals):.1f} бит")
    print(f"Медианный интервал: {np.median(intervals):.1f} бит")
    
    # Вычисляем коэффициент кластеризации
    cluster_threshold = window_size // 2
    clusters = []
    current_cluster = []
    
    for pos in positions:
        if not current_cluster or pos - current_cluster[-1] <= cluster_threshold:
            current_cluster.append(pos)
        else:
            clusters.append(current_cluster)
            current_cluster = [pos]
    
    if current_cluster:
        clusters.append(current_cluster)
    
    print(f"\nОбнаружено {len(clusters)} кластеров ошибок:")
    for i, cluster in enumerate(clusters[:5]):  # Показываем первые 5 кластеров
        print(f"Кластер {i+1}: {len(cluster)} ошибок между позициями {cluster[0]}-{cluster[-1]}")
    
    if len(clusters) > 5:
        print(f"... и еще {len(clusters)-5} кластеров")

def read_bits_from_file(filepath):
    """Чтение битовой последовательности из текстового файла с обработкой ошибок"""
    try:
        with open(filepath, 'r') as file:
            content = file.read().strip()
            if not all(c in '01' for c in content):
                print(f"⚠️ Внимание: файл {filepath} содержит не только биты (0/1)!")
            return content
    except FileNotFoundError:
        print(f"❌ Файл {filepath} не найден!")
        exit(1)
    except Exception as e:
        print(f"❌ Ошибка при чтении файла {filepath}: {str(e)}")
        exit(1)

def compare_bit_sequences(original, extracted):
    """Сравнение последовательностей по длине извлеченных битов"""
    extracted_len = len(extracted)
    if len(original) < extracted_len:
        print(f"⚠️ Внимание: оригинальная последовательность короче ({len(original)}) чем извлеченная ({extracted_len})")
        extracted = extracted[:len(original)]
        extracted_len = len(original)
    
    original_truncated = original[:extracted_len]
    errors = []
    
    for idx in range(extracted_len):
        if original_truncated[idx] != extracted[idx]:
            errors.append((idx, original_truncated[idx], extracted[idx]))
    
    return errors, extracted_len

def generate_report(errors, total_bits, output_dir):
    """Генерация текстового и графического отчета"""
    error_count = len(errors)
    error_percent = (error_count / total_bits) * 100 if total_bits > 0 else 0
    
    # Создаем директорию для отчетов, если ее нет
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(output_dir, f"bit_comparison_report_{timestamp}.txt")
    
    # Текстовый отчет
    with open(report_filename, 'w') as report_file:
        report_file.write("Отчет сравнения битовых последовательностей\n")
        report_file.write("="*50 + "\n")
        report_file.write(f"Сравнено бит: {total_bits}\n")
        report_file.write(f"Ошибочных бит: {error_count}\n")
        report_file.write(f"Процент ошибок: {error_percent:.6f}%\n\n")
        
        report_file.write("Первые 100 ошибок:\n")
        for idx, (pos, orig, extr) in enumerate(errors[:100]):
            report_file.write(f"{pos:8d} | было: {orig} → стало: {extr}\n")
            if idx == 99 and len(errors) > 100:
                report_file.write(f"\n... и еще {len(errors)-100} ошибок\n")
    
    # Визуализация
    if error_count > 0:
        plot_filename = os.path.join(output_dir, f"error_distribution_{timestamp}.png")
        plot_error_distribution([e[0] for e in errors], total_bits, plot_filename)
    
    return report_filename, error_percent

def plot_error_distribution(error_positions, total_bits, filename):
    """Визуализация распределения ошибок"""
    plt.figure(figsize=(12, 6))
    
    # Гистограмма распределения ошибок
    plt.subplot(1, 2, 1)
    bins = min(50, total_bits//100 or 1)  # Защита от деления на 0
    plt.hist(error_positions, bins=bins, color='red', alpha=0.7)
    plt.title('Распределение ошибок по позициям')
    plt.xlabel('Позиция бита')
    plt.ylabel('Количество ошибок')
    
    # Кумулятивная функция ошибок
    plt.subplot(1, 2, 2)
    plt.plot(sorted(error_positions), range(1, len(error_positions)+1), 'b-')
    plt.title('Накопление ошибок')
    plt.xlabel('Позиция бита')
    plt.ylabel('Количество ошибок')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"📊 График распределения ошибок сохранен в {filename}")

def plot_offset_analysis(original, extracted, start_pos=2000, max_offset=10):
    """Визуализация анализа смещения с выделением оптимального значения"""
    # Подготовка данных
    orig_tail = original[start_pos:]
    extr_tail = extracted[start_pos:]
    min_len = min(len(orig_tail), len(extr_tail))
    
    # Сбор статистики для разных смещений
    offsets = range(-max_offset, max_offset + 1)
    error_rates = []
    
    for offset in offsets:
        if offset < 0:
            compare_len = min_len + offset
            orig_part = orig_tail[-offset:]
            extr_part = extr_tail[:compare_len]
        else:
            compare_len = min_len - offset
            orig_part = orig_tail[:compare_len]
            extr_part = extr_tail[offset:]
        
        errors = sum(1 for o, e in zip(orig_part, extr_part) if o != e)
        error_rates.append((errors / compare_len) * 100 if compare_len > 0 else 100)
    
    # Находим оптимальное смещение
    best_idx = np.argmin(error_rates)
    best_offset = offsets[best_idx]
    best_rate = error_rates[best_idx]
    
    # Создание графика
    plt.figure(figsize=(12, 6))
    
    # Основной график
    ax = plt.subplot(1, 2, 1)
    bars = plt.bar(offsets, error_rates, color='skyblue')
    bars[best_idx].set_color('limegreen')
    
    # Выделяем оптимальное смещение
    plt.axvline(x=best_offset, color='red', linestyle='--', alpha=0.7)
    plt.text(best_offset, best_rate + 2, f'Оптимум: {best_offset}\n{best_rate:.2f}%', 
             ha='center', va='bottom', color='red')
    
    plt.title('Зависимость ошибок от смещения')
    plt.xlabel('Смещение (бит)')
    plt.ylabel('Процент ошибок (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # График сравнения последовательностей
    ax2 = plt.subplot(1, 2, 2)
    sample_len = min(100, min_len - abs(best_offset))
    
    if best_offset < 0:
        orig_sample = orig_tail[-best_offset:-best_offset+sample_len]
        extr_sample = extr_tail[:sample_len]
    else:
        orig_sample = orig_tail[:sample_len]
        extr_sample = extr_tail[best_offset:best_offset+sample_len]
    
    # Визуализация битовых последовательностей
    for i in range(sample_len):
        color = 'red' if orig_sample[i] != extr_sample[i] else 'green'
        plt.plot([i, i], [0, 1], color=color, linewidth=2)
    
    plt.title(f'Сравнение при смещении {best_offset}\n(красный = ошибка)')
    plt.xlabel('Позиция бита')
    plt.yticks([])
    plt.xlim(0, sample_len)
    
    plt.tight_layout()
    plt.savefig('best_offset_analysis.png', dpi=120)
    plt.close()
    
    print(f"\n✅ Оптимальное смещение: {best_offset} бит (ошибок: {best_rate:.2f}%)")
    print(f"📊 График сохранен в 'best_offset_analysis.png'")
    
    return best_offset

def main():
    # Конфигурация
    BASE_DIR = "./algorithmBMYeYu"
    REPORT_DIR = "./algorithmBMYeYu/reports"
    INPUT_FILE = os.path.join(BASE_DIR, "input_message_bit.txt")
    OUTPUT_FILE = os.path.join(BASE_DIR, "output_message_bit.txt")
    
    # Чтение данных
    print("\n🔎 Загрузка битовых последовательностей...")
    original_bits = read_bits_from_file(INPUT_FILE)
    extracted_bits = read_bits_from_file(OUTPUT_FILE)
    
    # Быстрая проверка
    print("\nℹ️ Контрольная информация:")
    print(f"Оригинальная длина: {len(original_bits)} бит")
    print(f"Извлеченная длина: {len(extracted_bits)} бит")
    print(f"\nОригинальная (первые 50): {original_bits[:50]}...")
    print(f"Извлеченная (первые 50): {extracted_bits[:50]}...")
    
    # Анализ (сравниваем по длине извлеченных битов)
    errors, compared_bits = compare_bit_sequences(original_bits, extracted_bits)
    report_file, error_rate = generate_report(errors, compared_bits, REPORT_DIR)
    
    
    # Вывод результатов
    print("\n📊 Результаты анализа:")
    print(f"• Сравнено бит: {compared_bits}")
    print(f"• Ошибок: {len(errors)}")
    print(f"• Процент ошибок: {error_rate:.6f}%")
    print(f"📄 Полный отчет сохранен в: {report_file}")
    
    # Оценка качества
    if len(errors) == 0:
        print("\n🎉 Отлично! Ошибок не обнаружено!")
    elif error_rate < 0.001:
        print("\n✅ Отличный результат! Минимальный уровень ошибок")
    elif error_rate < 1:
        print("\n⚠️ Умеренный уровень ошибок. Рекомендуется анализ")
    else:
        print("\n❌ Критический уровень ошибок! Требуется срочная доработка алгоритма")
    
    # Проверка работы на первых 1999 битах
    errors, _ = compare_bit_sequences(original_bits[:1999], extracted_bits[:1999])
    print(f"Ошибок в первых 1999 битах: {len(errors)}")

    # Анализ первых 2000 бит
    errors_first_2000, _ = compare_bit_sequences(original_bits[:2000], extracted_bits[:2000])
    print(f"\nПервые 2000 бит: {len(errors_first_2000)} ошибок ({len(errors_first_2000)/20:.1f}%)")

    # Анализ оставшейся части с поиском смещения
    best_offset = analyze_offset_errors(original_bits, extracted_bits)

    # Анализ кластеризации ошибок в "хвосте"
    if len(original_bits) > 2000 and len(extracted_bits) > 2000:
        tail_errors, compared_bits = compare_bit_sequences(
            original_bits[2000:], 
            extracted_bits[2000 + best_offset:] if best_offset > 0 else extracted_bits[2000:]
        )
        error_positions = [e[0] for e in tail_errors]
        analyze_error_clusters([p + 2000 for p in error_positions])

    best_offset = plot_offset_analysis(original_bits, extracted_bits)

    # Применение найденного смещения
    if best_offset != 0:
        corrected_bits = extracted_bits[best_offset:] if best_offset > 0 else extracted_bits[:best_offset]
        print(f"\nПрименено смещение {best_offset}, длина скорректированной последовательности: {len(corrected_bits)}")
if __name__ == "__main__":
    main()