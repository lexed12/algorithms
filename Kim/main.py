import cv2
import numpy as np
#from PIL import Image
import pywt
from typing import Dict, List, Tuple
from translate import *

alf3sa = 0.02
alf3 = 0.1
alf2 = 0.2
alf1 = 0.4

print('Hello World!')

def haar_2d_decomposition_with_C_S(image, levels=3):
    """
    Выполняет многоуровневое 2D вейвлет-разложение Хаара и возвращает аналоги `C` и `S` как в Matlab.
    
    Параметры:
        image (numpy.ndarray): Монохромное изображение (2D массив).
        levels (int): Количество уровней разложения (по умолчанию 3).
        
    Возвращает:
        tuple: (C, S, coeffs_dict)
            - C: 1D вектор всех коэффициентов (аналог Matlab `C`).
            - S: Массив с размерами подполос (аналог Matlab `S`).
            - coeffs_dict: Словарь с коэффициентами по уровням (как в предыдущей функции).
    """
    if len(image.shape) != 2:
        raise ValueError("Изображение должно быть монохромным (2D массив).")
    
    # Многоуровневое разложение
    coeffs = pywt.wavedec2(image, 'haar', level=levels)
    
    # Формируем C и S как в Matlab
    C, S = pywt.coeffs_to_array(coeffs)
    
    # Словарь с коэффициентами для удобства
    coeffs_dict = {
        'LL3': coeffs[0],  # sa3
        'LH3': coeffs[1][0],  # sh3
        'HL3': coeffs[1][1],  # sv3
        'HH3': coeffs[1][2],  # sd3
        'LH2': coeffs[2][0],  # sh2
        'HL2': coeffs[2][1],  # sv2
        'HH2': coeffs[2][2],  # sd2
        'LH1': coeffs[3][0],  # sh1
        'HL1': coeffs[3][1],  # sv1
        'HH1': coeffs[3][2],  # sd1
    }
    
    return C, S, coeffs

def find_top3_from_coeffs(coeffs):
    """
    Находит 3 максимальных значения (T1, T2, T3) среди коэффициентов разложения Хаара.
    
    Параметры:
        coeffs (list): Список коэффициентов, полученный из `pywt.wavedec2` или аналогичной функции.
                       Формат: [LL3, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1)].
        
    Возвращает:
        tuple: (T1, T2, T3) - три наибольших значения коэффициентов (по модулю).
    """
    # Собираем все коэффициенты в один массив
    all_coeffs = []
    for i, level in enumerate(coeffs):
        if i == 0:
            all_coeffs.append(level)  # LL3 (sa3)
        else:
            all_coeffs.extend([level[0], level[1], level[2]])  # LH, HL, HH
    
    # Объединяем в плоский массив и берём модуль
    flat_coeffs = np.abs(np.concatenate([c.flatten() for c in all_coeffs]))
    
    # Находим топ-3 максимума
    top3 = np.partition(flat_coeffs, -3)[-3:][::-1]  # Быстрее, чем полная сортировка
    С1, С2, С3 = top3[0], top3[1], top3[2]
    log2_T1 = np.log2(С1)
    T1 = 2 ** (log2_T1 - 1)
    log2_T2 = np.log2(С2)
    T2 = 2 ** (log2_T2 - 1)
    log2_T3 = np.log2(С3)
    T3 = 2 ** (log2_T3 - 1)
    return T1, T2, T3

def process_wavelet_coeffs(coeffs, thresholds, alpha, masCVZ):
    """
    Обрабатывает коэффициенты вейвлет-разложения с учетом разных alpha для sa3 и других коэффициентов.
    
    Параметры:
        coeffs (dict): Словарь коэффициентов {'sv3': массив, ..., 'sa3': массив}
        thresholds (dict): Пороги {'T3': float, 'T2': float, 'T1': float}
        alpha (dict): Коэффициенты {'alf3': float, 'alf2': float, 'alf1': float, 'alf3sa': float}
        masCVZ (np.array): Массив значений для модификации
    """
    masCVZ = [int(c) for c in masCVZ]
    message = [x if x != 0 else -1 for x in masCVZ]

    modified_coeffs = coeffs.copy()
    p = 0  # Индекс для masCVZ
    
    for level in [3, 2, 1]:
        for coeff_type in ['sv', 'sh', 'sd', 'sa']:
            coeff_name = f"{coeff_type}{level}"
            if coeff_name not in modified_coeffs:
                continue
                
            current_coeff = modified_coeffs[coeff_name]
            threshold = thresholds[f'T{level}']
            
            # Выбираем alpha в зависимости от типа коэффициента
            if coeff_name == 'sa3':  # Особый случай для LL3
                alpha_val = alpha['alf3sa']
            else:
                alpha_val = alpha[f'alf{level}']
            
            # Векторизованная обработка (быстрее чем вложенные циклы)
            mask = current_coeff > threshold
            if np.any(mask):
                modification = alpha_val * current_coeff[mask] * message[p:p+np.sum(mask)]
                current_coeff[mask] += modification
                p += np.sum(mask)
    
    return modified_coeffs
    
def list2dict_coeff(coeffs,
              T3,T2,T1):

    # Формируем словарь коэффициентов
    coeffs_dict = {
        'sa3': coeffs[0],     # LL3
        'sh3': coeffs[1][0],  # LH3
        'sv3': coeffs[1][1],  # HL3
        'sd3': coeffs[1][2],  # HH3
        'sh2': coeffs[2][0],  # LH2
        'sv2': coeffs[2][1],  # HL2
        'sd2': coeffs[2][2],  # HH2
        'sh1': coeffs[3][0],  # LH1
        'sv1': coeffs[3][1],  # HL1
        'sd1': coeffs[3][2],  # HH1
    }

    # Формируем словарь порогов
    thresholds = {
        'T3': T3,
        'T2': T2,
        'T1': T1
    }

    # Формируем словарь коэффициентов усиления
    alpha = {
        'alf3': 0.1,
        'alf2': 0.2,
        'alf1': 0.4,
        'alf3sa': 0.02  # Специальный коэффициент для sa3
    }

    # Размеры (если нужны)
    sizes = {
        'R3': coeffs[1][0].shape[0], 'RR3': coeffs[1][0].shape[1],
        'R2': coeffs[2][0].shape[0], 'RR2': coeffs[2][0].shape[1],
        'R1': coeffs[3][0].shape[0], 'RR1': coeffs[3][0].shape[1]
    }
    return coeffs_dict, thresholds, alpha, sizes


def fill_C_vector_fast(C, modified_coeffs):
    """
    Быстрое заполнение вектора C коэффициентами.
    Требует, чтобы C был одномерным массивом достаточной длины.
    """
    index = 0
    for coeff in ['sa3', 'sh3', 'sv3', 'sd3', 'sh2', 'sv2', 'sd2', 'sh1', 'sv1', 'sd1']:
        if coeff in modified_coeffs:
            data = modified_coeffs[coeff].flatten()
            end_idx = index + len(data)
            
            # Проверка переполнения
            if end_idx > len(C):
                raise ValueError(f"Недостаточно места в C для коэффициента {coeff}")
                
            C[index:end_idx] = data
            index = end_idx
    return C

def inverse_haar_transform(modified_coeffs, original_structure=None):
    """
    Выполняет обратное 2D DWT Хаара с проверкой структуры.
    
    Параметры:
        modified_coeffs (dict): Модифицированные коэффициенты 
        original_structure (list): Опционально - структура из pywt.wavedec2
        
    Возвращает:
        np.array: Восстановленное изображение
    """
    # Автоматическое определение структуры, если не указана
    if original_structure is None:
        coeffs_list = [
            modified_coeffs['sa3'],
            (modified_coeffs['sh3'], modified_coeffs['sv3'], modified_coeffs['sd3']),
            (modified_coeffs['sh2'], modified_coeffs['sv2'], modified_coeffs['sd2']),
            (modified_coeffs['sh1'], modified_coeffs['sv1'], modified_coeffs['sd1'])
        ]
    else:
        # Используем original_structure как шаблон
        coeffs_list = [modified_coeffs.get('sa3', original_structure[0])]
        for i in range(1, len(original_structure)):
            coeffs_list.append((
                modified_coeffs.get(f'sh{4-i}', original_structure[i][0]),
                modified_coeffs.get(f'sv{4-i}', original_structure[i][1]),
                modified_coeffs.get(f'sd{4-i}', original_structure[i][2])
            ))
    
    # Обратное преобразование
    reconstructed = pywt.waverec2(coeffs_list, 'haar')
    
    # Нормализация
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)


    return reconstructed

def encode(image, message):
    # Получаем C, S и коэффициенты
    C, S, coeffs = haar_2d_decomposition_with_C_S(image, levels=3)
    T1, T2, T3 = find_top3_from_coeffs(coeffs)
    R3, RR3 = coeffs[1][0].shape  # Получаем (число строк, число столбцов) coeffs[1][0] = sh3
    R2, RR2 = coeffs[2][0].shape  # Получаем (число строк, число столбцов) coeffs[2][0] = sh2
    R1, RR1 = coeffs[3][0].shape  # Получаем (число строк, число столбцов) coeffs[3][0] = sh1

    coeffs_dict, thresholds, alpha, sizes = list2dict_coeff(coeffs, T3, T2, T1)
    modified_coeffs = process_wavelet_coeffs(coeffs_dict, thresholds, alpha, message)
    embed = inverse_haar_transform(modified_coeffs, coeffs)
    return embed




def extract_watermark_all_fast(original_coeffs, modified_coeffs, thresholds, alpha):
    """
    Извлекает ЦВЗ из всех модифицированных коэффициентов
    
    Параметры:
        original_coeffs (dict): Исходные коэффициенты {'sa3': ..., 'sh3': ..., ...}
        modified_coeffs (dict): Модифицированные коэффициенты
        thresholds (dict): Пороги {'T1': float, 'T2': float, 'T3': float}
        alpha (dict): Коэффициенты усиления {'alf1': float, 'alf2': float, 'alf3': float, 'alf3sa': float}
        
    Возвращает:
        np.array: Объединенный вектор извлеченного ЦВЗ
    """
    coeff_pairs = [
        ('sa3', 'alf3sa', 'T3'),
        ('sh3', 'alf3', 'T3'),
        ('sv3', 'alf3', 'T3'),
        ('sd3', 'alf3', 'T3'),
        ('sh2', 'alf2', 'T2'),
        ('sv2', 'alf2', 'T2'),
        ('sd2', 'alf2', 'T2'),
        ('sh1', 'alf1', 'T1'),
        ('sv1', 'alf1', 'T1'),
        ('sd1', 'alf1', 'T1')
    ]
    
    watermark = []
    for coeff, alpha_key, threshold_key in coeff_pairs:
        orig = original_coeffs[coeff]
        mod = modified_coeffs[coeff]
        mask = (orig > thresholds[threshold_key]) & (orig != 0)
        wm_bits = (mod[mask] - orig[mask]) / (alpha[alpha_key] * orig[mask])
        watermark.append(wm_bits.flatten())
    
    return np.concatenate(watermark)

def binary_threshold(data):
    """
    Преобразует входные данные в бинарные значения:
    - Если элемент < 0 → 0
    - Если элемент >= 0 → 1
    
    Параметры:
        data: list или np.array
        
    Возвращает:
        np.array с бинарными значениями
    """
    arr = np.asarray(data)  # Конвертируем в массив NumPy
    return np.where(arr < 0, 0, 1).astype(np.uint8)


if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread('./Kim/les.jpg'), cv2.COLOR_BGR2GRAY)
    cv2.imshow('original1', image)
    message = text_to_bits(read_text_from_file('./Kim/input_message.txt'))
    embed = encode(image, message)


    C, S, coeffs = haar_2d_decomposition_with_C_S(image, levels=3)
    T1, T2, T3 = find_top3_from_coeffs(coeffs)
    R3, RR3 = coeffs[1][0].shape  # Получаем (число строк, число столбцов) coeffs[1][0] = sh3
    R2, RR2 = coeffs[2][0].shape  # Получаем (число строк, число столбцов) coeffs[2][0] = sh2
    R1, RR1 = coeffs[3][0].shape  # Получаем (число строк, число столбцов) coeffs[3][0] = sh1

    coeffs_dict, thresholds, alpha, sizes = list2dict_coeff(coeffs, T3, T2, T1)

    modified_C,  modified_S, mod_coeffs = haar_2d_decomposition_with_C_S(embed)
    # После встраивания ЦВЗ (как в предыдущих примерах)
    original_coeffs = {
        'sa3': coeffs[0], 'sh3': coeffs[1][0], 'sv3': coeffs[1][1], 'sd3': coeffs[1][2],
        'sh2': coeffs[2][0], 'sv2': coeffs[2][1], 'sd2': coeffs[2][2],
        'sh1': coeffs[3][0], 'sv1': coeffs[3][1], 'sd1': coeffs[3][2]
    }    
    modified_coeffs = {
        'sa3': mod_coeffs[0], 'sh3': mod_coeffs[1][0], 'sv3': mod_coeffs[1][1], 'sd3': mod_coeffs[1][2],
        'sh2': mod_coeffs[2][0], 'sv2': mod_coeffs[2][1], 'sd2': mod_coeffs[2][2],
        'sh1': mod_coeffs[3][0], 'sv1': mod_coeffs[3][1], 'sd1': mod_coeffs[3][2]
    }

    thresholds = {'T1': T1, 'T2': T2, 'T3': T3}
    alpha = {'alf1': alf1, 'alf2': alf2, 'alf3': alf3, 'alf3sa': alf3sa}

    extracted_watermark = extract_watermark_all_fast(original_coeffs, modified_coeffs, thresholds, alpha)
    bin_message = binary_threshold(extracted_watermark)


    cv2.imshow('embed', embed)

    


    rows, height = image.shape
    channels = 1
    cv2.waitKey(0)