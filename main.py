# main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
import pickle
import warnings

# Игнорируем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')


def filter_data(df):
    """
    Шаг 1: Удаление ненужных колонок (фильтрация датасета)
    """
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]

    # Возвращаем копию датафрейма, inplace тут делать нельзя!
    return df.drop(columns_to_drop, axis=1)


def handle_outliers(df):
    """
    Шаг 2: Сглаживание (удаление) выбросов в колонке year
    """

    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return boundaries

    # Работаем с копией датафрейма
    df_copy = df.copy()

    if 'year' in df_copy.columns:
        boundaries = calculate_outliers(df_copy['year'])
        df_copy.loc[df_copy['year'] < boundaries[0], 'year'] = round(boundaries[0])
        df_copy.loc[df_copy['year'] > boundaries[1], 'year'] = round(boundaries[1])

    return df_copy


def create_features(df):
    """
    Шаг 3: Создание новых предикторов (short_model и age_category)
    """

    def short_model(x):
        if not pd.isna(x):
            return x.lower().split(' ')[0]
        else:
            return x

    # Работаем с копией датафрейма
    df_copy = df.copy()

    # Добавляем фичу "short_model" – это первое слово из колонки model
    df_copy.loc[:, 'short_model'] = df_copy['model'].apply(short_model)

    # Добавляем фичу "age_category" (категория возраста)
    df_copy.loc[:, 'age_category'] = df_copy['year'].apply(
        lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average')
    )

    return df_copy


def main():
    print("Car Price Classification Pipeline")
    print("=" * 50)

    try:
        # Загрузка данных
        print("Загрузка данных...")
        df = pd.read_csv('data/homework.csv')
        print(f"Данные успешно загружены. Размер: {df.shape}")

        # Разделение на признаки и целевую переменную ДО препроцессинга
        X = df.drop('price_category', axis=1)
        y = df['price_category']

        print(f"Признаки: {X.shape}, Целевая переменная: {y.unique()}")

        # Создание ЕДИНОГО пайплайна со всеми шагами
        # Шаг 1: Удаление ненужных колонок
        # Шаг 2: Сглаживание выбросов в колонке year
        # Шаг 3: Создание новых предикторов
        preprocessing_pipeline = Pipeline(steps=[
            ('filter', FunctionTransformer(filter_data)),
            ('outlier_handling', FunctionTransformer(handle_outliers)),
            ('feature_engineering', FunctionTransformer(create_features))
        ])

        # Шаг 4: Заполнение пропусков в численных признаках медианой и масштабирование в StandardScaler
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Шаг 5: Заполнение пропусков в категориальных переменных и кодирование OneHotEncoder
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Объединение преобразователей для разных типов признаков
        feature_processor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
                ('cat', categorical_transformer, make_column_selector(dtype_include=object))
            ]
        )

        # Модели для сравнения (исправленная версия без liblinear)
        models = [
            LogisticRegression(solver='lbfgs', random_state=42, max_iter=1000),
            RandomForestClassifier(random_state=42),
            SVC(random_state=42)
        ]

        # Поиск лучшей модели на кросс-валидации (по метрике accuracy)
        best_score = 0
        best_pipeline = None
        best_model_name = ""

        print("\nОценка моделей на кросс-валидации:")
        print("-" * 50)

        for model in models:
            try:
                # Создание полного пайплайна со ВСЕМИ шагами
                pipe = Pipeline(steps=[
                    # Шаги 1-3: Предобработка данных
                    ('preprocessing', preprocessing_pipeline),
                    # Шаги 4-5: Обработка признаков
                    ('feature_processing', feature_processor),
                    # Итоговый шаг: Классификатор
                    ('classifier', model)
                ])

                # Кросс-валидация с метрикой accuracy
                scores = cross_val_score(pipe, X, y, cv=3, scoring='accuracy')  # Уменьшил cv для скорости
                mean_score = scores.mean()
                std_score = scores.std()

                print(f'model: {type(model).__name__:25} acc_mean: {mean_score:.4f}, acc_std: {std_score:.4f}')

                # Выбор лучшей модели
                if mean_score > best_score:
                    best_score = mean_score
                    best_pipeline = pipe
                    best_model_name = type(model).__name__

            except Exception as e:
                print(f"Ошибка в модели {type(model).__name__}: {e}")
                continue

        print("-" * 50)

        if best_pipeline is None:
            print("Ни одна модель не смогла обучиться!")
            return

        print(f'best model: {best_model_name}, accuracy: {best_score:.4f}')

        # Обучение лучшей модели на всех данных
        print(f"\nОбучение лучшей модели ({best_model_name})...")
        best_pipeline.fit(X, y)

        # Сохранение итогового пайплайна в pickle-файл
        pipeline_filename = 'car_price_pipeline.pkl'
        with open(pipeline_filename, 'wb') as f:
            pickle.dump(best_pipeline, f)

        print(f"✅ Пайплайн сохранен в файл '{pipeline_filename}'")

        # Проверка работы пайплайна
        try:
            predictions = best_pipeline.predict(X.head())
            print(f"\nПример предсказаний для первых 5 samples: {predictions}")
            print(f"Фактические значения: {y.head().values}")

            # Проверка точности
            train_accuracy = best_pipeline.score(X, y)
            print(f"Точность на обучающей выборке: {train_accuracy:.4f}")

        except Exception as e:
            print(f"Ошибка при проверке пайплайна: {e}")

        print("\n✅ Пайплайн завершен успешно!")

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    main()