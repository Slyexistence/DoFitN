import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import GridSearchCV
import joblib
import os

# Функция для замены бесконечных значений и обработки выбросов
def preprocess_features(df):
    # Замена inf значений на NaN и заполнение медианой
    df_processed = df.copy()
    
    # Заменяем inf на NaN
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Получаем список числовых колонок
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Для каждой числовой колонки заполняем NaN медианой
    for col in numeric_cols:
        if df_processed[col].isna().any():
            median_value = df_processed[col].median()
            df_processed[col].fillna(median_value, inplace=True)
            
    # Проверка наличия чрезмерно больших значений и их ограничение
    for col in numeric_cols:
        if col != 'label':  # Не обрабатываем целевую переменную
            q1 = df_processed[col].quantile(0.25)
            q3 = df_processed[col].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 3 * iqr
            
            # Ограничение очень больших значений
            df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
    
    return df_processed

def main():
    # Загрузка данных
    # Используем относительный путь
    data_path = r'D:\Проекты\Дипломаня работа\DoFitN\Code\DoFitN\Data\Data_20\combined_features.csv'
    print(f"Загрузка данных из {data_path}...")
    df = pd.read_csv(data_path)
    
    # Вывод информации о загруженных данных
    print(f"Размер датасета: {df.shape}")
    print(f"Распределение классов:\n{df['label'].value_counts(normalize=True)}")
    
    # Предобработка данных
    print("Предобработка данных...")
    df_processed = preprocess_features(df)
    
    # Удаление неинформативных столбцов
    columns_to_drop = ['timestamp', 'src_mac', 'dst_mac', 'src_ip', 'dst_ip', 'session_id']
    X = df_processed.drop(columns=columns_to_drop + ['label'])
    y = df_processed['label']
    
    print(f"Признаки для обучения: {X.columns.tolist()}")
    
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    
    # Определение числовых признаков для стандартизации
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Создание препроцессора для числовых признаков
    numeric_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())  # Используем RobustScaler для устойчивости к выбросам
    ])
    
    # Создание ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )
    
    # Определение архитектуры нейронной сети и параметров
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),  # Три скрытых слоя: 128, 64 и 32 нейрона
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 регуляризация
        batch_size=64,
        learning_rate='adaptive',
        max_iter=200,
        early_stopping=True,  # Использование раннего останова
        validation_fraction=0.1,  # 10% обучающих данных используется для валидации
        random_state=42,
        verbose=True
    )
    
    # Создание пайплайна
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', mlp)
    ])
    
    # Обучение модели
    print("Обучение модели...")
    pipeline.fit(X_train, y_train)
    
    # Оценка модели
    print("Оценка модели на тестовых данных...")
    y_pred = pipeline.predict(X_test)
    
    # Вычисление вероятностей для ROC кривой
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Вывод метрик
    print(f"\nТочность модели: {accuracy_score(y_test, y_pred):.4f}")
    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_pred))
    
    # Создание директории для сохранения результатов
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Визуализация результатов
    plt.figure(figsize=(12, 10))
    
    # Матрица ошибок
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Матрица ошибок')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    
    # ROC кривая
    plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC кривая')
    plt.legend(loc='lower right')
    
    # Кривая точности-полноты
    plt.subplot(2, 2, 3)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision)
    plt.xlabel('Полнота')
    plt.ylabel('Точность')
    plt.title('Кривая точности-полноты')
    
    # Важность признаков (для MLPClassifier сложно интерпретировать)
    plt.subplot(2, 2, 4)
    # Вместо важности признаков покажем распределение классов
    sns.countplot(x=y)
    plt.title('Распределение классов')
    plt.xlabel('Класс')
    plt.ylabel('Количество')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/mlp_results.png')
    print(f"Визуализация результатов сохранена в {results_dir}/mlp_results.png")
    
    # Сохранение модели
    model_path = f'{results_dir}/mlp_model.joblib'
    joblib.dump(pipeline, model_path)
    print(f"Модель сохранена в {model_path}")
    
    # Опционально: поиск оптимальных гиперпараметров
    # Раскомментируйте этот код, если хотите выполнить поиск гиперпараметров
    
    param_grid = {
        'classifier__hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
        'classifier__alpha': [0.0001, 0.001, 0.01],
        'classifier__learning_rate': ['constant', 'adaptive'],
        'classifier__activation': ['relu', 'tanh']
    }
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2
    )
    
    print("Поиск оптимальных гиперпараметров...")
    grid_search.fit(X_train, y_train)
    
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучшая F1-мера: {grid_search.best_score_:.4f}")
    
    # Сохранение лучшей модели
    best_model_path = f'{results_dir}/mlp_model_best.joblib'
    joblib.dump(grid_search.best_estimator_, best_model_path)
    print(f"Лучшая модель сохранена в {best_model_path}")
    

if __name__ == "__main__":
    main() 