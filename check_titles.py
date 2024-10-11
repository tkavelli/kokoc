import pandas as pd

# Путь к каталогу товаров
catalog_path = 'data/raw/stokman_catalog_preprocessed.pq'

# Функция для проверки наличия текстовых описаний товаров

def check_titles():
    try:
        # Загрузка данных каталога
        catalog = pd.read_parquet(catalog_path)

        # Проверка наличия столбца с текстовыми описаниями и количества непустых значений
        if 'title' in catalog.columns:
            non_empty_titles = catalog['title'].dropna().shape[0]
            total_items = catalog.shape[0]
            print(f"Количество товаров с текстовыми описаниями: {non_empty_titles} из {total_items}")

            # Вывод первых пяти текстовых описаний
            sample_titles = catalog['title'].dropna().head(5)
            print("\nПервые 5 текстовых описаний:")
            for i, title in enumerate(sample_titles, 1):
                print(f"{i}. {title}")
        else:
            print("Столбец 'title' отсутствует в данных каталога.")

    except FileNotFoundError:
        print(f"Файл {catalog_path} не найден. Проверьте путь к каталогу.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    check_titles()