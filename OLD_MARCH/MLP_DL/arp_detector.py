import os
import sys
import time
import numpy as np
import pandas as pd
from scapy.all import sniff, ARP
from collections import defaultdict
import logging
import argparse
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime

# Настройка логирования с датой и временем в имени файла
current_datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
log_filename = f"arp_detector_{current_datetime}.log"

# Сначала удаляем предыдущие обработчики, если они существуют
logger = logging.getLogger(__name__)
if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)

# Настройка логирования с новым именем файла
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, mode='w')  # режим 'w' для создания нового файла
    ]
)

# Добавляем маркер начала новой сессии
logger.info("=" * 50)
logger.info(f"НАЧАЛО НОВОЙ СЕССИИ: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
logger.info("=" * 50)

# Пути к моделям
MODEL_PATH = r"D:\Проекты\Дипломаня работа\Write_code\models\arp_spoofing_detector.keras"
SCALER_PATH = r"D:\Проекты\Дипломаня работа\Write_code\models\scaler.pkl"

class ARPSpoofingDetector:
    """Детектор ARP-spoofing атак на основе нейронной сети."""
    
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH, window_size=10, threshold=0.7):
        """
        Инициализирует детектор.
        
        Args:
            model_path: Путь к файлу модели
            scaler_path: Путь к файлу скейлера
            window_size: Размер окна отслеживания пакетов
            threshold: Порог для классификации пакета как аномального
        """
        self.window_size = window_size
        self.threshold = threshold
        
        # Загрузка модели и скейлера
        logger.info("Загрузка модели и скейлера...")
        try:
            self.model = load_model(model_path)
            logger.info(f"Модель загружена: {model_path}")
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Скейлер загружен: {scaler_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели или скейлера: {e}")
            raise
        
        # Статистика по ARP-трафику
        self.arp_stats = defaultdict(lambda: {
            'timestamps': [],
            'src_macs': [],
            'dst_macs': [],
            'opcodes': [],  # 1 для запроса, 2 для ответа
            'is_broadcasts': []
        })
        
        # Для отслеживания изменений MAC-адресов для IP
        self.ip_to_mac = {}
        # Счетчик подозрительных пакетов для каждого IP
        self.suspicious_count = defaultdict(int)
        # Счетчик ARP-запросов для обнаружения сканирования
        self.arp_scan_detection = {
            'start_time': time.time(),
            'request_count': 0
        }
        self.start_time = time.time()
        
        # фкяем структуры для более глубокого анализа ARP-трафика
        self.arp_replies_history = defaultdict(list)  # История ARP-ответов для каждого IP
        self.new_devices = {}  # Новые устройства в сети
        self.gateway_macs = set()  # MAC-адреса шлюзов
        self.trusted_devices = {}  # Доверенные устройства (IP-MAC пары)
        
        logger.info("Детектор ARP-spoofing атак инициализирован")
    
    def extract_features(self, packet):
        """Извлекает признаки из ARP-пакета."""
        arp = packet.getlayer(ARP)
        
        # Базовые признаки пакета
        src_mac = arp.hwsrc
        dst_mac = arp.hwdst
        src_ip = arp.psrc
        dst_ip = arp.pdst
        opcode = arp.op  # 1 для запроса, 2 для ответа
        is_broadcast = dst_mac == "ff:ff:ff:ff:ff:ff"
        
        # Обнаружение ARP-сканирования
        if opcode == 1:  # ARP-запрос
            self.arp_scan_detection['request_count'] += 1
            current_time = time.time()
            time_elapsed = current_time - self.arp_scan_detection['start_time']
            
            # Если более 50 запросов за 30 секунд, считаем это сканированием
            if time_elapsed <= 30 and self.arp_scan_detection['request_count'] > 50:
                logger.warning(f"Обнаружено ARP-сканирование: {self.arp_scan_detection['request_count']} запросов за {time_elapsed:.2f} секунд")
                
            # Сброс счетчика каждые 30 секунд
            if time_elapsed > 30:
                self.arp_scan_detection['start_time'] = current_time
                self.arp_scan_detection['request_count'] = 1
        
        # Ключ для хранения истории - пара IP-адресов
        key = f"{src_ip}_{dst_ip}"
        stats = self.arp_stats[key]
        current_time = time.time()
        
        # Обновление истории
        stats['timestamps'].append(current_time)
        stats['src_macs'].append(src_mac)
        stats['dst_macs'].append(dst_mac)
        stats['opcodes'].append(opcode)
        stats['is_broadcasts'].append(is_broadcast)
        
        # Ограничиваем размер истории
        if len(stats['timestamps']) > self.window_size:
            stats['timestamps'] = stats['timestamps'][-self.window_size:]
            stats['src_macs'] = stats['src_macs'][-self.window_size:]
            stats['dst_macs'] = stats['dst_macs'][-self.window_size:]
            stats['opcodes'] = stats['opcodes'][-self.window_size:]
            stats['is_broadcasts'] = stats['is_broadcasts'][-self.window_size:]
        
        # Выявление характеристик, которые могут указывать на ARP-spoofing
        duplicates = self._count_duplicates(stats)
        requests = stats['opcodes'].count(1)
        replies = stats['opcodes'].count(2)
        packet_rate = self._calculate_packet_rate(stats)
        multiple_macs = self._detect_multiple_macs(stats, src_ip, src_mac)
        request_reply_ratio = requests / replies if replies > 0 else float('inf')
        
        # Формируем признаки для предсказания
        features = pd.DataFrame([{
            'timestamp': current_time - self.start_time,
            'src_mac': src_mac,
            'dst_mac': dst_mac,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'opcode': opcode,
            'is_broadcast': is_broadcast,
            'duplicates': duplicates,
            'requests': requests,
            'replies': replies,
            'packet_rate': packet_rate,
            'multiple_macs': multiple_macs,
            'request_reply_ratio': request_reply_ratio
        }])
        
        return features, key
    
    def _count_duplicates(self, stats):
        """Подсчитывает количество дубликатов пакетов в окне."""
        if len(stats['src_macs']) <= 1:
            return 0
        
        duplicates = 0
        for i in range(len(stats['src_macs']) - 1):
            for j in range(i + 1, len(stats['src_macs'])):
                if (stats['src_macs'][i] == stats['src_macs'][j] and 
                    stats['dst_macs'][i] == stats['dst_macs'][j] and 
                    stats['opcodes'][i] == stats['opcodes'][j]):
                    duplicates += 1
        
        return duplicates
    
    def _calculate_packet_rate(self, stats):
        """Рассчитывает скорость пакетов в пакетах в секунду."""
        if len(stats['timestamps']) <= 1:
            return 0
        
        time_interval = stats['timestamps'][-1] - stats['timestamps'][0]
        if time_interval == 0:
            return 0
        
        return (len(stats['timestamps']) - 1) / time_interval
    
    def _detect_multiple_macs(self, stats, src_ip, src_mac):
        """Проверяет, связан ли один IP с несколькими MAC-адресами."""
        if src_ip in self.ip_to_mac and self.ip_to_mac[src_ip] != src_mac:
            logger.warning(f"Обнаружено изменение MAC-адреса для IP {src_ip}: {self.ip_to_mac[src_ip]} -> {src_mac}")
            # Увеличиваем счетчик подозрительных пакетов для этого IP
            self.suspicious_count[src_ip] += 1
            # Запоминаем изменение для этого IP
            self.ip_to_mac[src_ip] = src_mac
            # Возвращаем 1, что указывает на обнаружение потенциальной атаки
            return 1
        
        self.ip_to_mac[src_ip] = src_mac
        return 0
    
    def prepare_features(self, features_df):
        """Преобразует признаки для модели."""
        # Кодирование MAC-адресов
        for col in ['src_mac', 'dst_mac']:
            features_df[col] = features_df[col].str.replace(':', '').str.replace('-', '')
            features_df[f'{col}_int'] = features_df[col].apply(
                lambda x: int(x, 16) if pd.notna(x) and x != '' else 0
            )
            features_df.drop(col, axis=1, inplace=True)
        
        # Кодирование IP-адресов
        for col in ['src_ip', 'dst_ip']:
            features_df[f'{col}_int'] = features_df[col].apply(
                lambda x: sum(int(octet) * (256 ** (3 - i)) for i, octet in enumerate(x.split('.'))) 
                if pd.notna(x) and x != '0.0.0.0' and x != '' else 0
            )
            features_df.drop(col, axis=1, inplace=True)
        
        # Преобразование булевых признаков
        features_df['is_broadcast'] = features_df['is_broadcast'].astype(int)
        
        # Обработка бесконечных значений - исправление FutureWarning
        for col in features_df.columns:
            if (features_df[col] == np.inf).any() or (features_df[col] == -np.inf).any():
                # Устраняем предупреждение FutureWarning, используя рекомендуемый подход
                features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
                features_df[col] = features_df[col].fillna(10000)  # Заменяем на большое число
        
        # Применение скейлера
        numeric_features = features_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        features_df[numeric_features] = self.scaler.transform(features_df[numeric_features])
        
        return features_df
    
    def predict(self, features_df):
        """Выполняет предсказание на основе признаков пакета."""
        try:
            # Получение IP-адреса из признаков
            src_ip = features_df['src_ip'].values[0]
            dst_ip = features_df['dst_ip'].values[0]
            multiple_macs = features_df['multiple_macs'].values[0]
            
            # Проверка на явные признаки ARP spoofing
            
            # 1. Если обнаружена смена MAC-адреса для IP
            if multiple_macs == 1:
                # Это критический признак ARP-spoofing атаки
                logger.warning(f"ОБНАРУЖЕНА ARP-SPOOFING АТАКА: обнаружена смена MAC-адреса для IP {src_ip}")
                # Маркируем как высокую вероятность атаки
                return 0.95
                
            # 2. Проверка наличия подозрительных пакетов для src_ip
            if self.suspicious_count[src_ip] > 0:
                logger.warning(f"ПОДОЗРЕНИЕ НА ARP-SPOOFING: IP {src_ip} ранее менял MAC-адрес ({self.suspicious_count[src_ip]} раз)")
                return 0.90
            
            # 3. Проверка на gratuitous ARP (когда src_ip == dst_ip)
            if src_ip == dst_ip and src_ip != '0.0.0.0' and src_ip != '255.255.255.255':
                count = self.arp_stats.get(f"{src_ip}_{dst_ip}", {}).get('opcodes', []).count(2)  # Количество ответов
                if count > 3:  # Если было больше 3 таких ответов
                    logger.warning(f"ПОДОЗРЕНИЕ НА ARP-SPOOFING: обнаружен gratuitous ARP от {src_ip}")
                    return 0.85
            
            # Используем модель нейронной сети для предсказания на основе всех признаков
            X = self.prepare_features(features_df.copy())
            prediction = float(self.model.predict(X, verbose=0)[0][0])
            
            # Если предсказание выше порога, но не обнаружено явных признаков атаки,
            # проверяем другие факторы
            if prediction >= self.threshold:
                packet_rate = features_df['packet_rate'].values[0]
                duplicates = features_df['duplicates'].values[0]
                
                # Анализируем скорость пакетов и дубликаты для повышения уверенности
                if packet_rate > 10 or duplicates > 3:
                    logger.warning(f"ПОДОЗРЕНИЕ НА ARP-SPOOFING: высокая интенсивность ARP-пакетов от {src_ip}")
                    prediction = max(prediction, 0.80)
            
            return prediction
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            return 0.0
    
    def packet_callback(self, packet):
        """Обрабатывает пакет."""
        if packet.haslayer(ARP):
            arp = packet.getlayer(ARP)
            src_ip = arp.psrc
            dst_ip = arp.pdst
            src_mac = arp.hwsrc
            dst_mac = arp.hwdst
            opcode = arp.op
            
            # Проверяем подозрительные соответствия IP-MAC
            self._check_suspicious_ip_mac(src_ip, src_mac, opcode)
            
            features_df, key = self.extract_features(packet)
            probability = self.predict(features_df)
            
            # Определение типа ARP пакета
            packet_type = "запрос" if opcode == 1 else "ответ"
            
            # Логирование результата
            status = "[УГРОЗА]" if probability >= self.threshold else "[НОРМА]"
            log_message = f"{status} {src_ip} -> {dst_ip} | MAC: {src_mac} -> {dst_mac} | Тип: {packet_type} | Вероятность: {probability:.4f}"
            
            if probability >= self.threshold:
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            # Если обнаружена угроза, возможно здесь можно добавить дополнительные действия
            # (оповещение, блокирование и т.д.)
    
    def _check_suspicious_ip_mac(self, src_ip, src_mac, opcode):
        """
        Проверяет подозрительные соответствия IP-MAC и записывает историю ARP-ответов.
        
        Args:
            src_ip: IP-адрес источника
            src_mac: MAC-адрес источника
            opcode: Тип операции ARP (1 - запрос, 2 - ответ)
        """
        # Пропускаем специальные адреса
        if src_ip == '0.0.0.0' or src_ip == '255.255.255.255':
            return
        
        current_time = time.time()
        
        # Записываем ARP-ответы для анализа
        if opcode == 2:  # ARP-ответ
            self.arp_replies_history[src_ip].append({
                'mac': src_mac,
                'time': current_time
            })
            
            # Ограничиваем размер истории
            if len(self.arp_replies_history[src_ip]) > 20:
                self.arp_replies_history[src_ip] = self.arp_replies_history[src_ip][-20:]
            
            # Обнаружение множественных MAC-адресов в ответах для одного IP
            macs = set(entry['mac'] for entry in self.arp_replies_history[src_ip])
            if len(macs) > 1:
                mac_list = ', '.join(macs)
                logger.warning(f"ПОДОЗРЕНИЕ НА ARP-SPOOFING: IP {src_ip} ассоциирован с несколькими MAC-адресами: {mac_list}")
                self.suspicious_count[src_ip] += 1
            
            # Считаем шлюзами IP-адреса, оканчивающиеся на .1 или .254
            if src_ip.endswith('.1') or src_ip.endswith('.254'):
                if src_mac not in self.gateway_macs:
                    if self.gateway_macs:  # Если уже есть MAC-адреса шлюзов
                        logger.warning(f"ПОДОЗРЕНИЕ НА ARP-SPOOFING: Обнаружен новый MAC-адрес {src_mac} для потенциального шлюза {src_ip}")
                    self.gateway_macs.add(src_mac)
        
        # Обнаружение новых устройств в сети
        if src_ip not in self.new_devices:
            self.new_devices[src_ip] = {
                'mac': src_mac,
                'first_seen': current_time,
                'arp_replies': 0
            }
        else:
            # Если MAC изменился для нового устройства
            if self.new_devices[src_ip]['mac'] != src_mac:
                time_diff = current_time - self.new_devices[src_ip]['first_seen']
                # Если устройство появилось недавно (менее 5 минут) и уже сменило MAC
                if time_diff < 300:  # 5 минут в секундах
                    logger.warning(f"ПОДОЗРЕНИЕ НА ARP-SPOOFING: Новое устройство {src_ip} сменило MAC с "
                                f"{self.new_devices[src_ip]['mac']} на {src_mac} через {time_diff:.1f} секунд после появления")
                    self.suspicious_count[src_ip] += 2  # Более подозрительное поведение
                self.new_devices[src_ip]['mac'] = src_mac
            
            # Подсчет ARP-ответов от устройства
            if opcode == 2:
                self.new_devices[src_ip]['arp_replies'] += 1
                # Если новое устройство отправило много ARP-ответов за короткое время
                if self.new_devices[src_ip]['arp_replies'] > 10 and (current_time - self.new_devices[src_ip]['first_seen']) < 60:
                    logger.warning(f"ПОДОЗРЕНИЕ НА ARP-SPOOFING: Новое устройство {src_ip} ({src_mac}) отправило "
                                f"{self.new_devices[src_ip]['arp_replies']} ARP-ответов за менее чем 60 секунд")
    
    def start(self, interface=None, count=0):
        """Запускает сниффинг ARP-пакетов."""
        logger.info(f"Запуск детектора ARP-spoofing атак на интерфейсе: {interface if interface else 'все'}")
        logger.info(f"Порог обнаружения атак: {self.threshold}")
        logger.info("Для остановки нажмите Ctrl+C")
        
        try:
            sniff(
                filter="arp",
                prn=self.packet_callback,
                iface=interface,
                count=count,
                store=0
            )
        except KeyboardInterrupt:
            logger.info("Детектор остановлен пользователем")
        except Exception as e:
            logger.error(f"Ошибка при сниффинге: {e}")
            logger.exception("Подробная информация об ошибке:")

def parse_args():
    """Разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Детектор ARP-spoofing атак")
    parser.add_argument("-i", "--interface", help="Сетевой интерфейс для мониторинга (по умолчанию: все)")
    parser.add_argument("-t", "--threshold", type=float, default=0.7, 
                       help="Порог для классификации пакета как аномального (по умолчанию: 0.7)")
    parser.add_argument("-w", "--window", type=int, default=10,
                       help="Размер окна для анализа пакетов (по умолчанию: 10)")
    return parser.parse_args()

if __name__ == "__main__":
    # Разбор аргументов
    args = parse_args()
    
    try:
        # Создание и запуск детектора
        detector = ARPSpoofingDetector(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            window_size=args.window,
            threshold=args.threshold
        )
        
        detector.start(interface=args.interface)
        
    except Exception as e:
        logger.error(f"Ошибка при запуске детектора: {e}")
        logger.exception("Подробная информация об ошибке:")
        sys.exit(1) 