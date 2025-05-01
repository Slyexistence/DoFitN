'''
Промышленная система реального времени для обнаружения сетевой атаки ARP-spoofing 
на основе Scapy-сниффера, автоэнкодера и MLP-классификатора.

Что можно улучшить: сохранение логов в определённую директорию,
а не в текущую, и сделать так, чтобы они не перезаписывались, а добавлялись. 
Gui для настройки параметров и отображения логов. Также добавить алогоритмический 
подход к определению порога вероятности для обнаружения ARP-spoofing.


Система способна расширяться вертикально и горизонтально.

Вертикально:
- Добавление новых признаков.
- Добавление новых моделей классификации.
- Добавление новых методов обнаружения атак. 

Горизонтально:
- Добавление новых моделей классификации.
- Добавление новых методов обнаружения атак. 



'''

import scapy.all as scapy
import pandas as pd
import numpy as np
import hashlib
import joblib
import logging
import time
import datetime
import os
import ipaddress
import traceback
from collections import defaultdict
from tensorflow.keras.models import load_model, Model
from scipy.stats import entropy


# Настройка логирования

current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
log_filename = f"arp_detector_{current_time}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    encoding='utf-8'
)

# Добавляем вывод логов в консоль
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


# Пути к файлам моделей
# Определяем базовую директорию
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(script_dir, 'Models')

# Проверяем различные возможные пути к моделям
possible_base_dirs = [
    BASE_DIR,
    os.path.join(os.getcwd(), 'Models'),
    r'D:\Проекты\Дипломаня работа\DoFitN\Code\DoFitN\Job_version\Main_11_04_2025\Models',
    r'D:\Проекты\Дипломаня работа\DoFitN\Code\DoFitN\Main_11_04_2025\Models'
]

# Выбираем первый существующий путь
for path in possible_base_dirs:
    if os.path.exists(path):
        BASE_DIR = path
        break

# Формируем пути к файлам моделей
AUTOENCODER_PATH = os.path.join(BASE_DIR, 'autoencoder_model.keras')

# Проверяем различные возможные имена для MLP модели
mlp_possible_names = [
    'mlp_classifier.keras', 
    'mlp_classifier.joblib', 
    'mlp.joblib', 
    'mlp_model.joblib',
    'mlp_classifier.pkl',
    'mlp.pkl'
]

MLP_PATH = None
for mlp_name in mlp_possible_names:
    path = os.path.join(BASE_DIR, mlp_name)
    if os.path.exists(path):
        MLP_PATH = path
        break

if MLP_PATH is None:
    MLP_PATH = os.path.join(BASE_DIR, 'mlp_classifier.keras')  # Устанавливаем значение по умолчанию

# Путь к скейлеру
scaler_possible_names = ['scaler.pkl', 'scaler.joblib', 'robust_scaler.pkl']
SCALER_PATH = None
for scaler_name in scaler_possible_names:
    path = os.path.join(BASE_DIR, scaler_name)
    if os.path.exists(path):
        SCALER_PATH = path
        break

if SCALER_PATH is None:
    SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')  # Устанавливаем значение по умолчанию

logging.info(f"Загрузка моделей из: {BASE_DIR}")
print(f"Загрузка моделей из: {BASE_DIR}")
print(f"Автоэнкодер: {AUTOENCODER_PATH}")
print(f"MLP: {MLP_PATH}")
print(f"Скейлер: {SCALER_PATH}")


# Загрузка моделей

try:
    print("Загрузка автоэнкодера...")
    autoencoder = load_model(AUTOENCODER_PATH)
    
    print("Загрузка MLP...")
    try:
        # Сначала пробуем загрузить как модель Keras
        mlp = load_model(MLP_PATH)
        is_keras_model = True
    except:
        # Если не получилось, пробуем загрузить через joblib
        print("Попытка загрузки MLP через joblib...")
        mlp = joblib.load(MLP_PATH)
        is_keras_model = False
    
    print("Загрузка скейлера...")
    scaler = joblib.load(SCALER_PATH)
    
    # Вытаскиваем энкодер из автоэнкодера
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-2].output)
    
    print("Модели успешно загружены")
    logging.info("Модели успешно загружены")
except Exception as e:
    error_msg = f"Ошибка при загрузке моделей: {e}"
    logging.error(error_msg)
    logging.error(traceback.format_exc())
    print(error_msg)
    raise


# Ожидаемые признаки в том же порядке, как при обучении

EXPECTED_COLUMNS = [
    'timestamp', 'opcode', 'duplicates', 'requests', 'replies',
    'packet_rate', 'multiple_macs', 'request_reply_ratio',
    'time_since_last_packet', 'unique_ip_count', 'interval_entropy',
    'response_time', 'target_ip_diversity', 'is_gateway',
    'unanswered_requests', 'reply_percentage', 'log_packet_rate',
    'src_ip_hash', 'dst_ip_hash', 'src_mac_hash', 'dst_mac_hash'
]


# Функция хеширования IP и MAC

def hash_str(s, size=256):
    """Хеширование строки в целое число фиксированного размера"""
    if not s:
        return 0
    return int(hashlib.md5(str(s).encode()).hexdigest(), 16) % size


# Структуры данных для отслеживания ARP-активности

class NetworkTracker:
    def __init__(self):
        self.arp_history = defaultdict(list)      # История ARP-пакетов для каждого IP
        self.ip_mac_mappings = defaultdict(set)   # Связи IP-MAC
        self.mac_ip_mappings = defaultdict(set)   # Связи MAC-IP
        self.arp_request_times = defaultdict(list) # Времена ARP-запросов для каждого IP
        self.arp_reply_counts = defaultdict(int)   # Счетчики ARP-ответов
        self.mac_timestamps = defaultdict(list)    # Временные метки пакетов от MAC
        self.duplicate_counts = defaultdict(int)   # Счетчики дубликатов
        self.mac_last_seen = {}                    # Время последней активности MAC
        self.legitimate_mappings = set()           # Подтвержденные корректные пары IP-MAC
        self.start_time = time.time()              # Время начала работы системы

        # Окно времени для связывания запросов и ответов (в секундах)
        self.REQUEST_REPLY_WINDOW = 2.0

        # Настраиваемые параметры для снижения ложных срабатываний
        self.MULTIPLE_MAC_THRESHOLD = 2      # Сколько разных MAC для одного IP считать подозрительным
        self.MIN_PACKETS_FOR_DETECTION = 3   # Минимальное количество пакетов для анализа
        self.SPOOF_DETECTION_THRESHOLD = 0.85  # Порог вероятности для обнаружения ARP-spoofing


network_tracker = NetworkTracker()


# Функция извлечения признаков из ARP-пакета

def extract_arp_features(packet):
    try:
        if not packet.haslayer(scapy.ARP):
            return None
        
        # Извлекаем базовую информацию о пакете
        current_time = time.time()
        arp = packet[scapy.ARP]
        
        # Получаем IP и MAC адреса из ARP пакета
        src_ip = arp.psrc  # IP отправителя
        dst_ip = arp.pdst  # IP назначения
        src_mac = arp.hwsrc  # MAC отправителя
        dst_mac = arp.hwdst  # MAC назначения
        
        # Определяем тип ARP-пакета (1=запрос, 2=ответ)
        opcode = arp.op
        
        # Определяем, является ли запрос широковещательным
        is_broadcast = dst_mac == "00:00:00:00:00:00" or dst_mac == "ff:ff:ff:ff:ff:ff"
        
        # Обновляем связи IP-MAC
        if src_ip and src_mac:
            network_tracker.ip_mac_mappings[src_ip].add(src_mac)
            network_tracker.mac_ip_mappings[src_mac].add(src_ip)
            
            # Сохраняем историю ARP для этого IP
            network_tracker.arp_history[src_ip].append({
                'time': current_time,
                'mac': src_mac,
                'opcode': opcode,
                'is_broadcast': is_broadcast
            })
            
            # Ограничиваем размер истории
            if len(network_tracker.arp_history[src_ip]) > 100:
                network_tracker.arp_history[src_ip] = network_tracker.arp_history[src_ip][-100:]
        
        # Обновляем сведения о запросах и ответах
        if opcode == 1:  # ARP запрос
            # Для запроса сохраняем время запроса к целевому IP
            network_tracker.arp_request_times[dst_ip].append(current_time)
            # Ограничиваем историю запросов
            if len(network_tracker.arp_request_times[dst_ip]) > 20:
                network_tracker.arp_request_times[dst_ip] = network_tracker.arp_request_times[dst_ip][-20:]
        else:  # ARP ответ (opcode == 2)
            # Для ответа проверяем, был ли запрос к этому IP недавно
            recent_request = False
            for req_time in network_tracker.arp_request_times.get(src_ip, []):
                if current_time - req_time < network_tracker.REQUEST_REPLY_WINDOW:
                    recent_request = True
                    break
            
            # Если был запрос и это ответ, считаем это легитимной связью
            if recent_request:
                network_tracker.legitimate_mappings.add((src_ip, src_mac))
            
            # Увеличиваем счетчик ответов
            network_tracker.arp_reply_counts[(src_ip, dst_ip)] += 1
        
        # Обновляем временные метки для MAC
        network_tracker.mac_timestamps[src_mac].append(current_time)
        
        # Определяем дубликаты
        dup_key = (src_mac, src_ip, dst_ip)
        network_tracker.duplicate_counts[dup_key] += 1
        duplicates = network_tracker.duplicate_counts[dup_key] - 1  # -1 потому что первый пакет не дубликат
        
        # Вычисляем частоту пакетов
        packet_rate = 0
        if src_mac in network_tracker.mac_last_seen:
            time_diff = current_time - network_tracker.mac_last_seen[src_mac]
            if time_diff > 0:
                packet_rate = 1 / time_diff
        network_tracker.mac_last_seen[src_mac] = current_time
        
        # Посчитаем количество запросов и ответов для этой пары IP
        req_count = len(network_tracker.arp_request_times.get(dst_ip, []))
        rep_count = network_tracker.arp_reply_counts.get((src_ip, dst_ip), 0)
        
        # Вычисляем соотношение запросов/ответов
        request_reply_ratio = req_count / rep_count if rep_count > 0 else float('inf')
        
        # Вычисляем время с момента последнего пакета
        time_since_last_packet = 0
        mac_times = network_tracker.mac_timestamps[src_mac]
        if len(mac_times) > 1:
            sorted_times = sorted(mac_times)
            idx = sorted_times.index(current_time)
            if idx > 0:
                time_since_last_packet = current_time - sorted_times[idx-1]
        
        # Количество уникальных IP для этого MAC
        unique_ip_count = len(network_tracker.mac_ip_mappings[src_mac])
        
        # Энтропия интервалов
        interval_entropy = 0
        if len(mac_times) > 5:
            intervals = [mac_times[i+1] - mac_times[i] for i in range(len(mac_times)-1)]
            bins = np.histogram(intervals, bins=10)[0]
            if sum(bins) > 0:
                interval_entropy = entropy(bins+1)
        
        # Время ответа (используем окно времени для оценки)
        response_time = 0
        if opcode == 2:  # Это ответ
            # Ищем последний запрос к этому IP
            if network_tracker.arp_request_times.get(src_ip):
                latest_request = max(network_tracker.arp_request_times[src_ip])
                response_time = current_time - latest_request
        
        # Разнообразие целевых IP
        target_ip_diversity = len(set(ip for ip_list in network_tracker.arp_request_times.values() for ip in ip_list))
        
        # Является ли IP шлюзом
        is_gateway = 0
        try:
            ip_obj = ipaddress.IPv4Address(src_ip)
            if ip_obj.is_private:
                last_octet = int(src_ip.split('.')[-1])
                is_gateway = 1 if last_octet in (1, 254) else 0
        except:
            pass
        
        # Неотвеченные запросы и процент ответов
        unanswered_requests = max(0, req_count - rep_count)
        reply_percentage = (rep_count / req_count) * 100 if req_count > 0 else 0
        
        # Логарифмическая шкала для скорости пакетов
        log_packet_rate = np.log1p(packet_rate)
        
        # Признак множественных MAC для одного IP (ключевой признак ARP-spoofing!)
        # Учитываем только те MAC, которые не были подтверждены как легитимные
        suspicious_macs = set()
        for mac in network_tracker.ip_mac_mappings.get(src_ip, set()):
            if (src_ip, mac) not in network_tracker.legitimate_mappings:
                suspicious_macs.add(mac)
                
        multiple_macs = 1 if len(suspicious_macs) >= network_tracker.MULTIPLE_MAC_THRESHOLD else 0
        
        # Формируем признаки в том же порядке, как они были в датасете
        features_dict = {
            'timestamp': current_time - network_tracker.start_time,
            'opcode': opcode,
            'duplicates': duplicates,
            'requests': req_count,
            'replies': rep_count,
            'packet_rate': packet_rate,
            'multiple_macs': multiple_macs,
            'request_reply_ratio': request_reply_ratio,
            'time_since_last_packet': time_since_last_packet,
            'unique_ip_count': unique_ip_count,
            'interval_entropy': interval_entropy,
            'response_time': response_time,
            'target_ip_diversity': target_ip_diversity,
            'is_gateway': is_gateway,
            'unanswered_requests': unanswered_requests,
            'reply_percentage': reply_percentage,
            'log_packet_rate': log_packet_rate,
            'src_ip_hash': hash_str(src_ip),
            'dst_ip_hash': hash_str(dst_ip),
            'src_mac_hash': hash_str(src_mac),
            'dst_mac_hash': hash_str(dst_mac)
        }
        
        # Добавляем дополнительную информацию для анализа
        features_dict['is_broadcast'] = is_broadcast
        features_dict['has_recent_request'] = 1 if opcode == 2 and recent_request else 0
        features_dict['src_ip'] = src_ip
        features_dict['dst_ip'] = dst_ip
        features_dict['src_mac'] = src_mac
        features_dict['dst_mac'] = dst_mac
        
        return features_dict
    except Exception as e:
        logging.error(f"Ошибка при извлечении признаков ARP: {e}")
        logging.error(traceback.format_exc())
        return None


# Функция предсказания ARP-spoofing

def predict_arp_spoofing(features_dict):
    try:
        src_ip = features_dict.get('src_ip', '')
        src_mac = features_dict.get('src_mac', '')
        dst_ip = features_dict.get('dst_ip', '')
        dst_mac = features_dict.get('dst_mac', '')
        opcode = features_dict.get('opcode', 0)
        is_broadcast = features_dict.get('is_broadcast', False)
        has_recent_request = features_dict.get('has_recent_request', 0)
        
        # Игнорируем запросы с широковещательным MAC - это нормальное поведение
        if opcode == 1 and is_broadcast:
            return 0, 0.0, "Нормальный широковещательный ARP-запрос"
            
        # Игнорируем ответы, которые соответствуют недавним запросам
        if opcode == 2 and has_recent_request:
            return 0, 0.1, "Нормальный ARP-ответ на недавний запрос"
        
        # Проверяем, достаточно ли у нас данных для анализа
        if len(network_tracker.arp_history.get(src_ip, [])) < network_tracker.MIN_PACKETS_FOR_DETECTION:
            return 0, 0.1, "Недостаточно данных для анализа"
        
        # Проверка на несколько MAC для одного IP (ключевой признак ARP-spoofing)
        suspicious_macs = set()
        for mac in network_tracker.ip_mac_mappings.get(src_ip, set()):
            if (src_ip, mac) not in network_tracker.legitimate_mappings:
                suspicious_macs.add(mac)
                
        if len(suspicious_macs) >= network_tracker.MULTIPLE_MAC_THRESHOLD:
            spoofing_probability = 0.95
            reason = f"IP {src_ip} ассоциирован с несколькими MAC: {', '.join(suspicious_macs)}"
            return 1, spoofing_probability, reason
        
        # Проверка на ответы без запросов (только если это не легитимная связь)
        if opcode == 2 and not has_recent_request and (src_ip, src_mac) not in network_tracker.legitimate_mappings:
            # Проверяем более строго - может быть запрос был до запуска сниффера
            # Если это первые несколько пакетов, не считаем это подозрительным
            if len(network_tracker.arp_history.get(src_ip, [])) > network_tracker.MIN_PACKETS_FOR_DETECTION:
                spoofing_probability = 0.9
                reason = f"Получены ARP-ответы без запросов для {src_ip}"
                return 1, spoofing_probability, reason
        
        # Используем модель машинного обучения для сложных случаев
        
        # Создаем DataFrame с признаками в правильном порядке
        feature_values = []
        for col in EXPECTED_COLUMNS:
            if col in features_dict:
                feature_values.append(features_dict[col])
            else:
                feature_values.append(0)
        
        # Преобразуем список в numpy array для скейлера
        features_array = np.array([feature_values])
        
        # Заменяем бесконечности
        features_array = np.nan_to_num(features_array, nan=0, posinf=0, neginf=0)
        
        # Масштабирование признаков
        scaled_features = scaler.transform(features_array)
        
        # Извлечение латентных признаков через энкодер
        latent_features = encoder.predict(scaled_features, verbose=0)
        
        # Получаем вероятности классов в зависимости от типа модели
        if is_keras_model:
            # Для модели Keras
            probabilities = mlp.predict(latent_features, verbose=0)
            # Вероятность класса "аномалия"
            if probabilities.shape[1] > 1:
                spoofing_probability = float(probabilities[0][1])  # Вероятность класса 1 (аномалия)
            else:
                spoofing_probability = float(probabilities[0][0])  # Одно значение: вероятность аномалии
        else:
            # Для scikit-learn модели
            try:
                # Если модель поддерживает predict_proba
                probabilities = mlp.predict_proba(latent_features)
                spoofing_probability = float(probabilities[0][1])  # Вероятность класса 1 (аномалия)
            except:
                # Если модель не поддерживает predict_proba, используем predict
                prediction_binary = mlp.predict(latent_features)
                spoofing_probability = 1.0 if prediction_binary[0] == 1 else 0.0
        
        # Применяем порог
        prediction = 1 if spoofing_probability >= network_tracker.SPOOF_DETECTION_THRESHOLD else 0
        reason = "Обнаружено на основе модели машинного обучения"
        
        return prediction, spoofing_probability, reason
    except Exception as e:
        logging.error(f"Ошибка при предсказании ARP-spoofing: {e}")
        logging.error(traceback.format_exc())
        return 0, 0.0, f"Ошибка анализа: {e}"


# Основная функция обработки ARP-пакетов

def arp_packet_callback(packet):
    # Извлекаем признаки из ARP-пакета
    features_dict = extract_arp_features(packet)
    
    # Если признаки успешно извлечены
    if features_dict is not None:
        # Получаем важные значения для логирования
        src_ip = features_dict['src_ip']
        dst_ip = features_dict['dst_ip']
        src_mac = features_dict['src_mac']
        dst_mac = features_dict['dst_mac']
        opcode = features_dict['opcode']
        is_broadcast = features_dict['is_broadcast']
        
        # Определяем ARP-spoofing
        prediction, probability, reason = predict_arp_spoofing(features_dict)
        
        # Формируем сообщение
        arp_type = "ARP-запрос" if opcode == 1 else "ARP-ответ"
        if is_broadcast:
            arp_type += " (широковещательный)"
            
        status = "АТАКА ARP-SPOOFING" if prediction == 1 else "Норма"
        
        log_message = (
            f"{arp_type} от {src_ip} ({src_mac}) к {dst_ip} ({dst_mac}) "
            f"-> {status} (вероятность: {probability:.4f})"
        )
        
        # Если это атака, добавляем причину
        if prediction == 1:
            log_message += f"\nПричина: {reason}"
            
            # Дополнительная информация для анализа
            known_macs = list(network_tracker.ip_mac_mappings.get(src_ip, set()))
            if len(known_macs) > 1:
                log_message += f"\nIP {src_ip} связан с MAC-адресами: {', '.join(known_macs)}"
        
        # Логируем и выводим результат
        logging.info(log_message)
        print(log_message)


# Запуск ARP-снифера

if __name__ == "__main__":
    try:
        print(f"ARP-сниффер запущен для обнаружения ARP-spoofing... Нажмите Ctrl+C для остановки.")
        print(f"Логи сохраняются в: {log_filename}")
        print(f"Настройки: порог множества MAC={network_tracker.MULTIPLE_MAC_THRESHOLD}, мин. пакетов={network_tracker.MIN_PACKETS_FOR_DETECTION}")
        logging.info(f"ARP-сниффер запущен с настройками: порог={network_tracker.SPOOF_DETECTION_THRESHOLD}, мин.пакетов={network_tracker.MIN_PACKETS_FOR_DETECTION}")
        
        # Фильтруем только ARP-пакеты
        scapy.sniff(filter="arp", prn=arp_packet_callback, store=False)
    except KeyboardInterrupt:
        print("\nARP-сниффер остановлен.")
        logging.info("ARP-сниффер остановлен пользователем.")
    except Exception as e:
        error_msg = f"Произошла ошибка: {e}"
        print(error_msg)
        logging.error(error_msg)
        logging.error(traceback.format_exc())