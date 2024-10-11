import numpy as np

# Путь к файлу vectors.npz
file_path = 'E:/kokoc/data/raw/vectors.npz'

# Загружаем файл vectors.npz
try:
    data = np.load(file_path, allow_pickle=True)

    # Выводим список всех ключей в файле
    print("Ключи в файле vectors.npz:", data.files)

    # Извлекаем данные под ключом 'arr_0'
    if 'arr_0' in data:
        arr_0 = data['arr_0']
        print("Тип данных под ключом 'arr_0':", type(arr_0))
        
        # Проверяем, являются ли данные массивом и выводим первые 10 элементов
        if isinstance(arr_0, np.ndarray):
            print("Первые 10 элементов из 'arr_0':")
            for i, item in enumerate(arr_0[:10]):
                print(f"{i + 1}: {item}")
        else:
            print("Данные под ключом 'arr_0' не являются массивом.")
    else:
        print("Ключ 'arr_0' не найден.")

except Exception as e:
    print(f"Ошибка при загрузке файла vectors.npz: {e}")