# Инструкция по запуску
1. Структура проекта
flower-similarity/
├── notebooks/
│   └── main.ipynb         # Предобработка данных, обучение модели, тестирование
├── app/
│   └── app.py                           # Flask API
├── models/
│   └── feature_extractor.h5             # Сохранённая модель-экстрактор признаков
├── embeddings/
│   ├── embeddings.npy                   # Предвычисленные эмбеддинги тестовой выборки
│   └── image_paths.pkl                  # Пути к изображениям из тестовой выборки
├── flowers/                             # Исходный датасет
│   ├── daisy/
│   ├── dandelion/
│   └── ...
├── Dockerfile                           # Для сборки Docker-образа
├── requirements.txt                     # Зависимости Python
└── README.md                            # Это описание

2. Для запуска
Скопировать репозиторий
git clone https://github.com/yourusername/flower-similarity.git 
cd flower-similarity

Собрать Docker контейнер
docker build -t flowers-similarity .

Запустить контейнер
docker run -p 5000:5000 -d flowers-similarity

3. Пример использования
- Postman
- curl
- Python-скрипт

# Содержание репозитория

🖼️ Визуализация результатов
В Jupyter Notebook вы найдете код, который:

Выбирает 5 случайных изображений из тестовой выборки
Отображает входное изображение и топ-5 совпадений в виде сетки из 6 изображений

🐳 Деплой в Docker
Функциональность
Модель загружается из файла models/feature_extractor.h5
Эмбеддинги и пути к изображениям загружаются из embeddings/
API принимает файл изображения и возвращает JSON с топ-5 результатами

🛠️ Проверка работы
1. Я лично проверял работу приложения в контейнере через POSTMAN и /app/test_request.py
2. Ответ в обоих случаях приходит в виде:
{'../dataset/flowers\\sunflower\\9206376642_8348ba5c7a.jpg': 0.7, '../dataset/flowers\\sunflower\\5139969871_c9046bdaa7_n.jpg': 0.69, '../dataset/flowers\\sunflower\\5979669004_d9736206c9_n.jpg': 0.68, '../dataset/flowers\\sunflower\\40411100_7fbe10ec0f_n.jpg': 0.66, '../dataset/flowers\\sunflower\\5979111555_61b400c070_n.jpg': 0.65}