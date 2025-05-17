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
- Скопировать репозиторий
``` git clone https://github.com/up99/flowers_similarity.git ```
``` cd flower-similarity ```

- Собрать Docker контейнер
``` docker build -t flowers-similarity . ```

- Запустить контейнер
``` docker run -p 5000:5000 -d flowers-similarity ```

3. Пример использования
- Postman
- curl
``` curl -X POST -F "file=@dataset/flowers/daisy/5547758_eea9edfd54_n.jpg" http://localhost:5000/predict ```
- Python-скрипт
```python3 app/test_request.py```


# Содержание репозитория

🖼️ Визуализация результатов
В Jupyter Notebook вы найдете код, который:

Выбирает 5 случайных изображений из тестовой выборки, отображает входное изображение и топ-5 совпадений в виде сетки

🐳 Деплой в Docker
Модель загружается из файла models/feature_extractor.h5

Эмбеддинги и пути к изображениям загружаются из embeddings/ и image_paths/ соответственно

API принимает файл изображения и возвращает JSON с топ-5 результатами

🛠️ Проверка работы
1. Я лично проверял работу приложения в контейнере через POSTMAN и /app/test_request.py
2. Ответ в обоих случаях приходит в виде:

**{'../dataset/flowers\\sunflower\\9206376642_8348ba5c7a.jpg': 0.7, '../dataset/flowers\\sunflower\\5139969871_c9046bdaa7_n.jpg': 0.69, '../dataset/flowers\\sunflower\\5979669004_d9736206c9_n.jpg': 0.68, '../dataset/flowers\\sunflower\\40411100_7fbe10ec0f_n.jpg': 0.66, '../dataset/flowers\\sunflower\\5979111555_61b400c070_n.jpg': 0.65}**

# Возникшие проблемы при решении задания
1. Функция jsonify()
- Долго не мог найти способ сохранить порядок отсортированных similarity scores для изображений. На сервере получал правильные результаты, то клиенту приходил ответ в JSON, гле нарушался порядок сортировки. Решилось с помощью app.json.sort_keys = False
2. Docker контейнер
- Долго не мог собрать контейнер из-за (как часто это бывает) знаков препинания: были перепутаны точки и двоеточия, сборка падала.
- Помимо этого, были конфликты пакетов. Особенно из-за TensorFlow и Keras. Удалось поправить не сразу. 

# Подход к решению задачи
1. Transfer learning
- Задача распространённая, поэтому есть множество готовых похожих проектов.
- Выбор в пользу популярных сетей (ResNet) - есть как в torch, так и в TF / Keras
- В итоге выбрал TF / Keras, так как удобнее инструменты для предобработки данных и подготовки датасета
2. StackOverFlow + LLM 