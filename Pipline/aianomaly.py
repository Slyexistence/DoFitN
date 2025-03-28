import os
import random
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter


os.environ["HF_DATASETS_CACHE"] = r"D:\cash"

# Путь к файлу с данными (каждая строка содержит две колонки, разделённые табуляцией: topic и message)
DATA_FILE = r"D:\Проекты\Дипломаня работа\DoFitN\Code\DoFitN\new_code\data_csv\combined_features.csv"
SAVE_PATH = r"D:\Проекты\Дипломаня работа\DoFitN\Code\DoFitN\Pipline\save"

# Загружаем датасет с использованием ключа "train"
dataset = load_dataset("csv", data_files={"train": DATA_FILE}, delimiter="\t", column_names=["topic", "message"])

# Получаем список уникальных тем (типов мошенничества)
topics = list(set(dataset["train"]["topic"]))
topics.sort()  # для стабильности
print("Уникальные типы мошенничества:", topics)

# Создаем словарь для преобразования темы в числовой идентификатор
topic2id = {topic: idx for idx, topic in enumerate(topics)}
id2topic = {idx: topic for topic, idx in topic2id.items()}

def map_labels(example):
    example["label"] = topic2id[example["topic"]]
    return example

dataset = dataset["train"].map(map_labels)

# Добавляем промпт для задания контекста и объединяем в поле "text"
def add_prompt(example):
    prompt = ("Определите, к какому типу мошенничества относится следующее сообщение. "
              "Возможные варианты: " + ", ".join(topics) + ". Сообщение:")
    example["text"] = f"{prompt} {example['message']}"
    return example

dataset = dataset.map(add_prompt)

# Перемешиваем датасет для случайного распределения
dataset = dataset.shuffle(seed=42)
print("Распределение меток:", Counter(dataset["label"]))

# Разбиваем датасет на обучающую и тестовую выборки (80/20)
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
data_dict = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"]
})

# Выбираем предобученную модель и токенизатор для русского языка
model_checkpoint = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
num_labels = len(topics)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# Функция токенизации (используем поле "text")
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = data_dict.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Функция вычисления метрик: accuracy и взвешенный F1
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Экспериментальные гиперпараметры
training_args = TrainingArguments(
    output_dir="./finetune_fraud_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1.5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
)

# Создаем Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Запуск обучения (Trainer автоматически использует GPU, если доступен)
trainer.train()

# Оценка модели на тестовой выборке
results = trainer.evaluate()
print("Evaluation results:", results)

# Сохранение дообученной модели и токенизатора
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"Model and tokenizer saved in {SAVE_PATH}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Получаем предсказания на тестовой выборке
predictions_output = trainer.predict(tokenized_datasets["test"])
pred_labels = np.argmax(predictions_output.predictions, axis=1)
true_labels = predictions_output.label_ids

# Выводим матрицу ошибок
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=topics)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Матрица ошибок классификации (Confusion Matrix)")
plt.tight_layout()
plt.show()

# Выводим топ ошибок
from collections import defaultdict
errors = defaultdict(list)
for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
    if true != pred:
        true_name = id2topic[true]
        pred_name = id2topic[pred]
        msg = data_dict["test"][i]["message"]
        errors[(true_name, pred_name)].append(msg)

# Показываем 3 примера самых частых ошибок
print("\n🔍 Топ ошибок классификации:\n")
sorted_errors = sorted(errors.items(), key=lambda x: len(x[1]), reverse=True)
for (true_name, pred_name), msgs in sorted_errors[:3]:
    print(f"❌ Истинная метка: {true_name} → Предсказано как: {pred_name} ({len(msgs)} раз)")
    for m in msgs[:3]:  # Покажем только 3 примера, чтобы не захламлять
        print("   ➤", m)
    print()
