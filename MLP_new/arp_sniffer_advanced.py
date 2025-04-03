#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import numpy as np
import pandas as pd
from scapy.all import sniff, ARP
from collections import defaultdict
import logging
import argparse
import joblib
import socket
import struct
from scipy import stats
from datetime import datetime

# Настройка логирования с датой и временем в имени файла
current_datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
log_filename = f"arp_detector_{current_datetime}.log"

# Сначала удаляем предыдущие обработчики, если они существуют
logger = logging.getLogger()
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

# Путь к модели MLP
MODEL_PATH = r'D:\Проекты\Дипломаня работа\results\mlp_model_best.joblib'
ALT_MODEL_PATH = r'results/mlp_model.joblib'

def ip2int(addr):
    """Преобразование IP-адреса в целое число"""
    try:
        return struct.unpack("!I", socket.inet_aton(addr))[0]
    except:
        return 0

def calculate_entropy(values):
    """Рассчитывает энтропию значений"""
    # Проверяем, является ли values массивом NumPy
    if isinstance(values, np.ndarray):
        # Если это пустой массив или массив с одним элементом
        if values.size == 0 or values.size == 1:
            return 0.0
        
        # Преобразуем NumPy массив в список для дальнейшей обработки
        values_list = values.tolist()
    else:
        # Для обычных списков Python
        if not values or len(values) <= 1:
            return 0.0
        values_list = values
    
    # Рассчитываем частотность уникальных значений
    value_counts = {}
    for value in values_list:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1
    
    # Рассчитываем энтропию
    total = len(values_list)
    entropy = 0.0
    for count in value_counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    
    return entropy

class ARPSpoofingDetector:
    """Детектор ARP-spoofing атак на основе MLP-модели."""
    
    def __init__(self, model_path=MODEL_PATH, window_size=10, threshold=0.7):
        """
        Инициализирует детектор.
        
        Args:
            model_path: Путь к файлу модели MLP
            window_size: Размер окна отслеживания пакетов
            threshold: Порог для классификации пакета как аномального
        """
        self.window_size = window_size
        self.threshold = threshold
        
        # Загрузка модели
        logger.info("Загрузка модели...")
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Модель успешно загружена: {model_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            try:
                logger.info(f"Пробуем загрузить модель из альтернативного пути: {ALT_MODEL_PATH}")
                self.model = joblib.load(ALT_MODEL_PATH)
                logger.info("Модель успешно загружена из альтернативного пути")
            except Exception as e2:
                logger.error(f"Не удалось загрузить модель: {e2}")
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
        
        # Структуры для более глубокого анализа ARP-трафика
        self.arp_replies_history = defaultdict(list)  # История ARP-ответов для каждого IP
        self.new_devices = {}  # Новые устройства в сети
        self.gateway_macs = set()  # MAC-адреса шлюзов
        self.trusted_devices = {}  # Доверенные устройства (IP-MAC пары)
        
        # Для хранения истории сессий
        self.sessions = {}
        self.session_features = {}
        self.session_counter = 0
        
        # Для хранения всех известных IP-адресов
        self.known_ips = set()
        
        # Для хранения последних пакетов по IP
        self.last_packet_time = {}
        
        # Для учета запросов/ответов
        self.request_response_pairs = defaultdict(list)
        
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
        
        # Добавляем IP в список известных
        self.known_ips.add(src_ip)
        self.known_ips.add(dst_ip)
        
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
        
        # Создаем сессионный ключ
        session_key = f"{src_ip}_{dst_ip}_{src_mac}_{dst_mac}"
        
        # Если это новая сессия, назначаем ей ID
        if session_key not in self.sessions:
            self.sessions[session_key] = self.session_counter
            self.session_counter += 1
            self.session_features[session_key] = {
                "timestamp": time.time(),
                "src_mac": src_mac,
                "dst_mac": dst_mac,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "session_id": self.sessions[session_key],
                "packet_count": 0,
                "request_count": 0, 
                "reply_count": 0,
                "interval_mean": 0,
                "interval_std": 0,
                "interval_min": 0,
                "interval_max": 0,
                "src_unique_count": 1,
                "dst_unique_count": 1,
                "ip_unique_count": 2,  # начинаем с src_ip и dst_ip
                "mac_unique_count": 2,  # начинаем с src_mac и dst_mac
                "unique_src_ips": set([src_ip]),
                "unique_dst_ips": set([dst_ip]),
                "unique_src_macs": set([src_mac]),
                "unique_dst_macs": set([dst_mac]),
                # Добавляем недостающие признаки
                "requests": 0,
                "replies": 0,
                "duplicates": 0,
                "opcode": opcode,
                "is_gateway": 0,  # По умолчанию не шлюз
                "unanswered_requests": 0,
                "reply_percentage": 0,
                "packet_rate": 0,
                "log_packet_rate": 0,
                "response_time": 0,
                "time_since_last_packet": 0,
                "interval_entropy": 0,
                "target_ip_diversity": 0,
                "unique_ip_count": 2  # Начинаем с src_ip и dst_ip
            }
            
        # Обновляем количество пакетов и типы операций
        self.session_features[session_key]["packet_count"] += 1
        if opcode == 1:  # ARP запрос
            self.session_features[session_key]["request_count"] += 1
            self.session_features[session_key]["requests"] += 1
            
            # Отслеживаем запросы для учета неотвеченных
            req_key = f"{src_ip}_{dst_ip}"
            self.request_response_pairs[req_key].append({
                "time": time.time(),
                "answered": False
            })
        elif opcode == 2:  # ARP ответ
            self.session_features[session_key]["reply_count"] += 1
            self.session_features[session_key]["replies"] += 1
            
            # Отмечаем, что на запрос был получен ответ
            req_key = f"{dst_ip}_{src_ip}"  # Обратный порядок для ответа
            found = False
            for req in self.request_response_pairs[req_key]:
                if not req["answered"]:
                    req["answered"] = True
                    # Вычисляем время ответа
                    self.session_features[session_key]["response_time"] = time.time() - req["time"]
                    found = True
                    break
        
        # Ключ для хранения истории - пара IP-адресов
        stats_key = f"{src_ip}_{dst_ip}"
        stats = self.arp_stats[stats_key]
        current_time = time.time()
        
        # Вычисляем время с момента последнего пакета для этого IP
        if src_ip in self.last_packet_time:
            self.session_features[session_key]["time_since_last_packet"] = current_time - self.last_packet_time[src_ip]
        self.last_packet_time[src_ip] = current_time
        
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
        
        # Обновляем сессионные данные
        # Обновляем данные о уникальных адресах в session_features
        self.session_features[session_key]["unique_src_ips"].add(src_ip)
        self.session_features[session_key]["unique_dst_ips"].add(dst_ip)
        self.session_features[session_key]["unique_src_macs"].add(src_mac)
        self.session_features[session_key]["unique_dst_macs"].add(dst_mac)
        
        # Обновляем счетчики уникальных адресов
        self.session_features[session_key]["src_unique_count"] = len(self.session_features[session_key]["unique_src_ips"])
        self.session_features[session_key]["dst_unique_count"] = len(self.session_features[session_key]["unique_dst_ips"])
        
        all_ips = set()
        all_ips.update(self.session_features[session_key]["unique_src_ips"])
        all_ips.update(self.session_features[session_key]["unique_dst_ips"])
        self.session_features[session_key]["ip_unique_count"] = len(all_ips)
        self.session_features[session_key]["unique_ip_count"] = len(all_ips)
        
        all_macs = set()
        all_macs.update(self.session_features[session_key]["unique_src_macs"])
        all_macs.update(self.session_features[session_key]["unique_dst_macs"])
        self.session_features[session_key]["mac_unique_count"] = len(all_macs)
        
        # Расчет интервалов между пакетами
        if session_key not in self.arp_stats:
            self.arp_stats[session_key] = []
        
        self.arp_stats[session_key].append(current_time)
        
        # Обновляем признаки интервалов
        if len(self.arp_stats[session_key]) > 1:
            intervals = np.diff(self.arp_stats[session_key])
            self.session_features[session_key]["interval_mean"] = np.mean(intervals)
            self.session_features[session_key]["interval_std"] = np.std(intervals) if len(intervals) > 1 else 0
            self.session_features[session_key]["interval_min"] = np.min(intervals)
            self.session_features[session_key]["interval_max"] = np.max(intervals)
            
            # Энтропия интервалов
            self.session_features[session_key]["interval_entropy"] = calculate_entropy(intervals)
        
        # Дополнительные признаки
        duration = max(0.001, time.time() - self.session_features[session_key]["timestamp"])
        
        # Скорость поступления пакетов
        packet_rate = self.session_features[session_key]["packet_count"] / duration
        self.session_features[session_key]["packet_rate"] = packet_rate
        self.session_features[session_key]["log_packet_rate"] = np.log1p(packet_rate)  # log(1+x) для избегания log(0)
        self.session_features[session_key]["arp_rate"] = packet_rate
        
        # Подсчет дубликатов пакетов
        if len(stats['timestamps']) > 1:
            duplicates = 0
            for i in range(len(stats['src_macs']) - 1):
                for j in range(i + 1, len(stats['src_macs'])):
                    if (stats['src_macs'][i] == stats['src_macs'][j] and 
                        stats['dst_macs'][i] == stats['dst_macs'][j] and 
                        stats['opcodes'][i] == stats['opcodes'][j]):
                        duplicates += 1
            self.session_features[session_key]["duplicates"] = duplicates
        
        # Диверсификация целевых IP (насколько разнообразны целевые IP)
        target_ips = list(self.session_features[session_key]["unique_dst_ips"])
        if len(target_ips) > 1:
            # Мера диверсификации: используем нормализованную энтропию целевых IP
            self.session_features[session_key]["target_ip_diversity"] = calculate_entropy(target_ips) / math.log2(len(target_ips))
        else:
            self.session_features[session_key]["target_ip_diversity"] = 0
        
        # Избегаем деления на ноль
        if self.session_features[session_key]["reply_count"] > 0:
            self.session_features[session_key]["request_reply_ratio"] = self.session_features[session_key]["request_count"] / self.session_features[session_key]["reply_count"]
            self.session_features[session_key]["reply_percentage"] = self.session_features[session_key]["reply_count"] / (self.session_features[session_key]["request_count"] + self.session_features[session_key]["reply_count"]) * 100
        else:
            self.session_features[session_key]["request_reply_ratio"] = self.session_features[session_key]["request_count"]  # Если ответов нет, то соотношение равно количеству запросов
            self.session_features[session_key]["reply_percentage"] = 0
        
        # Подсчет неотвеченных запросов
        unanswered = 0
        for key, reqs in self.request_response_pairs.items():
            if key.startswith(f"{src_ip}_"):
                unanswered += sum(1 for req in reqs if not req["answered"])
                
        self.session_features[session_key]["unanswered_requests"] = unanswered
        
        # Конвертируем IP в числовой формат для использования в модели
        self.session_features[session_key]["src_ip_int"] = ip2int(src_ip)
        self.session_features[session_key]["dst_ip_int"] = ip2int(dst_ip)
        
        # Выявление изменений MAC-адресов для IP (ключевой признак ARP-spoofing)
        multiple_macs = 0
        if src_ip in self.ip_to_mac and self.ip_to_mac[src_ip] != src_mac:
            logger.warning(f"Обнаружено изменение MAC-адреса для IP {src_ip}: {self.ip_to_mac[src_ip]} -> {src_mac}")
            self.suspicious_count[src_ip] += 1
            multiple_macs = 1
        self.ip_to_mac[src_ip] = src_mac
        
        # Добавляем этот признак в session_features
        self.session_features[session_key]["multiple_macs"] = multiple_macs
        
        # Определяем, является ли адрес шлюзом (обычно IP заканчивается на .1 или .254)
        if src_ip.endswith('.1') or src_ip.endswith('.254'):
            self.session_features[session_key]["is_gateway"] = 1
        
        # Проверка на gratuitous ARP (когда src_ip == dst_ip)
        is_gratuitous = src_ip == dst_ip and src_ip != '0.0.0.0' and src_ip != '255.255.255.255'
        self.session_features[session_key]["is_gratuitous"] = int(is_gratuitous)
        
        # Возвращаем признаки текущей сессии и ключ сессии
        return self.session_features[session_key], session_key
    
    def preprocess_features(self, features):
        """Предобработка признаков для модели MLP."""
        # Создаем копию признаков без множеств, которые не могут быть сериализованы в DataFrame
        features_copy = {k: v for k, v in features.items() 
                       if not isinstance(v, set)}
        
        df = pd.DataFrame([features_copy])
        
        # Удаляем неинформативные столбцы, как при обучении модели
        cols_to_drop = ['timestamp', 'src_mac', 'dst_mac', 'src_ip', 'dst_ip', 'session_id']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Заменяем бесконечные значения на NaN и заполняем медианными значениями
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Заполняем NaN медианными значениями (или нулями, если медиану нельзя вычислить)
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
        
        return df
    
    def predict(self, features, session_key):
        """Выполняет предсказание на основе признаков пакета."""
        try:
            # Выполняем преобработку признаков
            df = self.preprocess_features(features)
            
            # Получаем IP и MAC адреса для отчетов
            src_ip = features["src_ip"]
            dst_ip = features["dst_ip"]
            src_mac = features["src_mac"]
            dst_mac = features["dst_mac"]
            multiple_macs = features.get("multiple_macs", 0)
            
            # Проверка на явные признаки ARP spoofing
            manual_detection = False
            detection_reason = ""
            probability = 0.0
            
            # 1. Если обнаружена смена MAC-адреса для IP
            if multiple_macs == 1:
                manual_detection = True
                detection_reason = f"обнаружена смена MAC-адреса для IP {src_ip}"
                probability = 0.95
                
            # 2. Проверка наличия подозрительных пакетов для src_ip
            elif self.suspicious_count[src_ip] > 3:
                manual_detection = True
                detection_reason = f"IP {src_ip} многократно менял MAC-адрес ({self.suspicious_count[src_ip]} раз)"
                probability = 0.90
            
            # 3. Проверка на gratuitous ARP (когда src_ip == dst_ip)
            elif features.get("is_gratuitous", 0) == 1:
                count = sum(1 for t in self.arp_stats.get(f"{src_ip}_{dst_ip}", {}).get('opcodes', []) if t == 2)
                if count > 3:  # Если было больше 3 таких ответов
                    manual_detection = True
                    detection_reason = f"обнаружен множественный gratuitous ARP от {src_ip}"
                    probability = 0.85
            
            # Если нет явных признаков атаки, используем модель MLP
            if not manual_detection:
                # Используем модель для предсказания
                try:
                    # Выводим колонки для отладки
                    logger.debug(f"Имеющиеся колонки: {df.columns.tolist()}")
                    
                    # Используем безопасный метод получения предсказания
                    prediction_result = self.model.predict(df)
                    prediction = prediction_result[0] if hasattr(prediction_result, '__getitem__') else prediction_result
                    
                    # Получаем вероятность атаки
                    proba_result = self.model.predict_proba(df)
                    if hasattr(proba_result, '__getitem__'):
                        first_element = proba_result[0]
                        if hasattr(first_element, '__getitem__'):
                            probability = first_element[1]
                        else:
                            probability = first_element
                    else:
                        probability = proba_result
                    
                    # Если вероятность выше порога
                    if probability >= self.threshold:
                        # Дополнительные проверки для подтверждения
                        packet_rate = features.get("packet_rate", 0)
                        packet_count = features.get("packet_count", 0)
                        
                        if packet_count >= 5 and packet_rate > 1.0:
                            manual_detection = True
                            detection_reason = f"высокая интенсивность ARP-трафика: {packet_rate:.2f} пакетов/сек"
                        else:
                            manual_detection = True
                            detection_reason = f"модель классифицировала трафик как подозрительный"
                except Exception as e:
                    logger.error(f"Ошибка при использовании модели: {e}")
                    import traceback
                    traceback.print_exc()
                    return probability, False, ""
            
            return probability, manual_detection, detection_reason
        
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, False, ""
    
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
            
            # Извлекаем признаки пакета
            features, session_key = self.extract_features(packet)
            
            # Проверяем, достаточно ли собрано признаков для анализа
            if features["packet_count"] >= 3:
                # Получаем прогноз и информацию об обнаружении
                probability, is_attack, reason = self.predict(features, session_key)
                
                # Определение типа ARP пакета
                packet_type = "запрос" if opcode == 1 else "ответ"
                
                # Логирование результата
                status = "[УГРОЗА]" if is_attack else "[НОРМА]"
                log_message = f"{status} {src_ip} -> {dst_ip} | MAC: {src_mac} -> {dst_mac} | Тип: {packet_type} | Вероятность: {probability:.4f}"
                
                # Если это атака, добавляем причину обнаружения
                if is_attack:
                    log_message += f" | Причина: {reason}"
                    logger.warning(log_message)
                    print("\n[ВНИМАНИЕ] Обнаружен возможный ARP-spoofing!")
                    print(f"Вероятность атаки: {probability:.4f}")
                    print(f"Причина: {reason}")
                    print(f"Источник: {src_ip} ({src_mac})")
                    print(f"Назначение: {dst_ip} ({dst_mac})")
                    print(f"Количество пакетов: {features['packet_count']}")
                    print(f"Соотношение запросов/ответов: {features['request_reply_ratio']:.2f}")
                    print("=" * 60)
                else:
                    logger.info(log_message)
                    if features["packet_count"] % 10 == 0:  # Показываем информацию о нормальном трафике реже
                        print(f"\n[НОРМАЛЬНО] ARP трафик ({src_ip} -> {dst_ip})")
                        print(f"Вероятность атаки: {probability:.4f}")
                        print("-" * 40)
            else:
                # Просто выводим информацию о пакете
                packet_type = "запрос" if opcode == 1 else "ответ"
                print(f"ARP {packet_type}: {src_ip} ({src_mac}) -> {dst_ip} ({dst_mac})")
                
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
    parser = argparse.ArgumentParser(description="Детектор ARP-spoofing атак на основе MLP")
    parser.add_argument("-i", "--interface", help="Сетевой интерфейс для мониторинга (по умолчанию: все)")
    parser.add_argument("-t", "--threshold", type=float, default=0.7, 
                       help="Порог для классификации пакета как аномального (по умолчанию: 0.7)")
    parser.add_argument("-w", "--window", type=int, default=10,
                       help="Размер окна для анализа пакетов (по умолчанию: 10)")
    parser.add_argument("-m", "--model", help="Путь к модели (по умолчанию: используется предопределенный путь)")
    return parser.parse_args()

if __name__ == "__main__":
    # Разбор аргументов
    args = parse_args()
    
    try:
        # Определение пути к модели
        model_path = args.model if args.model else MODEL_PATH
        
        # Создание и запуск детектора
        detector = ARPSpoofingDetector(
            model_path=model_path,
            window_size=args.window,
            threshold=args.threshold
        )
        
        detector.start(interface=args.interface)
        
    except Exception as e:
        logger.error(f"Ошибка при запуске детектора: {e}")
        logger.exception("Подробная информация об ошибке:")
        sys.exit(1) 