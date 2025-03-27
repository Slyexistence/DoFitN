import argparse
import os
import time
import numpy as np
import pandas as pd
from scapy.all import *
from scapy.layers.l2 import ARP, Ether
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque, defaultdict
import threading
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime
from scapy.all import sniff, get_if_list
warnings.filterwarnings('ignore')

class BehavioralARPSniffer:
    def __init__(self, threshold=0.17, window_size=10, interface=None, learning_period=30):
        """
        Инициализация сниффера ARP-пакетов на основе поведенческого анализа
        
        Параметры:
        -----------
        threshold (float): Порог ошибки реконструкции для определения аномалий
        window_size (int): Размер временного окна в секундах для анализа
        interface (str): Сетевой интерфейс для сканирования
        learning_period (int): Период обучения для новых устройств в секундах
        """
        print("[+] Инициализация сниффера ARP-атак с поведенческим анализом...")
        self.model_path = r"D:\Проекты\Дипломаня работа\DoFitN\Code\DoFitN\new_code\Test\game_models\AE_game\arp_spoofing_detector.h5"
        self.threshold = threshold
        self.window_size = window_size
        self.interface = interface
        self.learning_period = learning_period
        
        # Создание директорий для логов
        self.logs_dir = "logs"
        self.sessions_dir = os.path.join(self.logs_dir, "sessions")
        self.alerts_dir = os.path.join(self.logs_dir, "alerts")
        
        for directory in [self.logs_dir, self.sessions_dir, self.alerts_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Файлы для логирования
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = os.path.join(self.sessions_dir, f"session_{timestamp}.log")
        self.alert_file = os.path.join(self.alerts_dir, f"alerts_{timestamp}.log")
        
        # Создаем файлы с заголовками
        with open(self.session_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Сессия мониторинга ARP-трафика (Поведенческий анализ) ===\n")
            f.write(f"Начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Интерфейс: {interface if interface else 'все'}\n")
            f.write(f"Размер окна: {window_size} сек\n")
            f.write(f"Порог обнаружения: {threshold}\n")
            f.write(f"Период обучения для новых устройств: {learning_period} сек\n")
            f.write("="*50 + "\n\n")
            
        with open(self.alert_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Лог обнаруженных атак ===\n")
            f.write(f"Начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
        
        # Проверка существования модели
        if not os.path.exists(self.model_path):
            print(f"[-] Ошибка: Модель не найдена по пути: {self.model_path}")
            sys.exit(1)
            
        # Инициализация буфера пакетов
        self.packet_buffer = deque(maxlen=1000)
        
        # Отслеживание поведения устройств
        self.device_profiles = {}  # Хранение профилей устройств
        self.device_first_seen = {}  # Время первого появления устройства
        self.device_history = defaultdict(list)  # История ошибок реконструкции
        self.mac_ip_mappings = {}  # Отслеживание сопоставлений MAC-IP
        self.ip_mac_mappings = {}  # Отслеживание сопоставлений IP-MAC
        
        # Белый список известных устройств
        self.whitelist = {
            # формат: 'MAC': 'описание'
            '00:ad:24:bf:9d:52': 'Маршрутизатор 192.168.1.1',
        }
        
        # Загрузка модели
        self.model = None
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.model.compile(optimizer='adam', loss='mse')
            print(f"[+] Автоэнкодер загружен из {self.model_path}")
        except Exception as e:
            print(f"[-] Ошибка при загрузке модели: {str(e)}")
            sys.exit(1)
        
        # Создание скейлера для нормализации данных
        self.scaler = StandardScaler()
        
        print("[+] Сниффер инициализирован и готов к запуску")
    
    def mac_to_int(self, mac):
        """Преобразует MAC-адрес в малое нормализованное число"""
        if isinstance(mac, str):
            try:
                # Используем только последний байт MAC-адреса
                last_byte = int(mac.split(':')[-1], 16)
                return last_byte / 255.0  # Нормализация до диапазона [0, 1]
            except:
                return 0
        return 0

    def ip_to_int(self, ip):
        """Преобразует IP-адрес в малое нормализованное число"""
        if isinstance(ip, str):
            try:
                # Используем только последний байт IP-адреса
                last_byte = int(ip.split('.')[-1])
                return last_byte / 255.0  # Нормализация до диапазона [0, 1]
            except:
                return 0
        return 0

    def ip_to_int(self, ip):
        """Преобразует IP-адрес в число с нормализацией"""
        if isinstance(ip, str):
            try:
                parts = ip.split('.')
                raw_value = int(parts[0]) * 16777216 + int(parts[1]) * 65536 + int(parts[2]) * 256 + int(parts[3])
                return raw_value / 10**9  # Масштабирование до приемлемого диапазона
            except:
                return 0
        return 0
    
    def is_suspicious_behavior(self, src_mac, features_dict, reconstruction_error):
        """Определяет, является ли поведение устройства подозрительным"""
        current_time = time.time()
        
        # Проверка, новое ли это устройство
        if src_mac not in self.device_first_seen:
            self.device_first_seen[src_mac] = current_time
            self.device_history[src_mac] = []
            print(f"[+] Новое устройство: {src_mac} - начало периода обучения ({self.learning_period} сек)")
            return False, "Новое устройство в периоде обучения"
        
        # Проверка, находится ли устройство в периоде обучения
        time_known = current_time - self.device_first_seen[src_mac]
        if time_known < self.learning_period:
            # В периоде обучения сохраняем ошибки, но не считаем поведение подозрительным
            self.device_history[src_mac].append(reconstruction_error)
            return False, f"Устройство в периоде обучения ({int(time_known)}/{self.learning_period} сек)"
        
        # После периода обучения проверяем подозрительное поведение
        # Добавляем текущую ошибку в историю устройства
        self.device_history[src_mac].append(reconstruction_error)
        
        # Ограничиваем историю последними 20 значениями
        if len(self.device_history[src_mac]) > 20:
            self.device_history[src_mac] = self.device_history[src_mac][-20:]
        
        # Рассчитываем среднюю и стандартное отклонение ошибок реконструкции для этого устройства
        device_errors = np.array(self.device_history[src_mac])
        mean_error = np.mean(device_errors)
        std_error = np.std(device_errors)
        
        # Устанавливаем динамический порог для конкретного устройства
        device_threshold = mean_error + 2 * std_error
        
        # Проверяем поведенческие аномалии
        is_suspicious = False
        reason = "Нормальное поведение"
        
        # 1. Проверка резкого изменения ошибки реконструкции
        if reconstruction_error > max(device_threshold, self.threshold):
            is_suspicious = True
            reason = f"Аномальная ошибка реконструкции: {reconstruction_error:.6f}"
        
        # 2. Проверка на подозрительное количество запросов/ответов
        if features_dict['packet_rate'] > 5 and features_dict['duplicates'] > 3:
            is_suspicious = True
            reason = f"Высокая частота пакетов ({features_dict['packet_rate']:.2f}/с) и дубликаты ({features_dict['duplicates']})"
        
        # 3. Проверка на изменение IP-MAC привязок
        if src_mac in self.device_profiles:
            last_ip = self.device_profiles[src_mac].get('last_ip')
            current_ip = features_dict.get('src_ip_num')
            
            if last_ip and current_ip and last_ip != current_ip:
                is_suspicious = True
                reason = f"Изменение IP-адреса устройства: {last_ip} -> {current_ip}"
        
        # Обновляем профиль устройства
        self.device_profiles[src_mac] = {
            'last_seen': current_time,
            'last_ip': features_dict.get('src_ip_num'),
            'avg_error': mean_error,
            'threshold': device_threshold
        }
        
        return is_suspicious, reason
    
    def calculate_window_features(self, window_packets):
        """Вычисляет признаки на основе окна пакетов"""
        if not window_packets:
            return None
        
        # Вычисление признаков
        total_packets = len(window_packets)
        duplicates = 0
        requests = 0
        replies = 0
        broadcast_count = 0
        unicast_count = 0
        
        # Счетчики для различных типов пакетов
        seen_packets = set()
        
        # Расчет времени между пакетами
        timestamps = [p['timestamp'] for p in window_packets]
        time_diffs = []
        if len(timestamps) > 1:
            for i in range(1, len(timestamps)):
                time_diffs.append(timestamps[i] - timestamps[i-1])
                
        # Для расчета частоты пакетов
        time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 1
        
        for packet in window_packets:
            packet_hash = f"{packet['src_mac']}_{packet['dst_mac']}_{packet['src_ip']}_{packet['dst_ip']}_{packet['opcode']}"
            
            # Проверка на дубликаты
            if packet_hash in seen_packets:
                duplicates += 1
            else:
                seen_packets.add(packet_hash)
            
            # Счетчики запросов и ответов ARP
            if packet['opcode'] == 1:  # request
                requests += 1
            elif packet['opcode'] == 2:  # reply
                replies += 1
                
            # Счетчики broadcast и unicast пакетов
            if packet['is_broadcast']:
                broadcast_count += 1
            else:
                unicast_count += 1
                
            # Отслеживание соответствий MAC-IP
            src_mac = packet['src_mac']
            src_ip = packet['src_ip']
            
            # Обновляем карту соответствий MAC-IP
            if src_mac not in self.mac_ip_mappings:
                self.mac_ip_mappings[src_mac] = set()
            self.mac_ip_mappings[src_mac].add(src_ip)
            
            # Обновляем карту соответствий IP-MAC
            if src_ip not in self.ip_mac_mappings:
                self.ip_mac_mappings[src_ip] = set()
            self.ip_mac_mappings[src_ip].add(src_mac)
        
        # Расчет статистики
        multiple_macs = len(set(p['src_mac'] for p in window_packets)) > 1
        request_reply_ratio = requests / replies if replies > 0 else requests
        packet_rate = total_packets / time_span if time_span > 0 else 0
        
        # Подсчет множественных IP-адресов для каждого MAC и наоборот
        mac_with_multi_ip = sum(1 for mac, ips in self.mac_ip_mappings.items() if len(ips) > 1)
        ip_with_multi_mac = sum(1 for ip, macs in self.ip_mac_mappings.items() if len(macs) > 1)
        
        # Формируем вектор признаков для автоэнкодера
        features_dict = {
            'timestamp': time.time(),
            'src_mac_num': self.mac_to_int(window_packets[-1]['src_mac']),
            'dst_mac_num': self.mac_to_int(window_packets[-1]['dst_mac']),
            'src_ip_num': self.ip_to_int(window_packets[-1]['src_ip']),
            'dst_ip_num': self.ip_to_int(window_packets[-1]['dst_ip']),
            'opcode': window_packets[-1]['opcode'],
            'is_broadcast': int(window_packets[-1]['is_broadcast']),
            'duplicates': duplicates,
            'requests': requests,
            'replies': replies,
            'packet_rate': packet_rate,
            'multiple_macs': int(multiple_macs),
            'request_reply_ratio': request_reply_ratio,
            'src_mac': window_packets[-1]['src_mac'],  # Добавляем исходный MAC для отслеживания
            'src_ip': window_packets[-1]['src_ip'],    # Добавляем исходный IP для отслеживания
            'mac_with_multi_ip': mac_with_multi_ip,   # Кол-во MAC с несколькими IP
            'ip_with_multi_mac': ip_with_multi_mac    # Кол-во IP с несколькими MAC
        }
        
        return features_dict
    
    def preprocess_features(self, features_dict):
        """Преобразование словаря признаков в формат для автоэнкодера"""
        # Выбираем только те признаки, которые использовались при обучении модели
        model_features = {
            'timestamp': 0,  # Заменяем на 0 для стабильности
            'src_mac_num': features_dict.get('src_mac_num', 0),
            'dst_mac_num': features_dict.get('dst_mac_num', 0),
            'src_ip_num': features_dict.get('src_ip_num', 0),
            'dst_ip_num': features_dict.get('dst_ip_num', 0),
            'opcode': features_dict.get('opcode', 0) / 2.0,  # Нормализация (1 или 2)
            'is_broadcast': features_dict.get('is_broadcast', 0),  # Уже 0 или 1
            'duplicates': min(features_dict.get('duplicates', 0) / 10.0, 1.0),  # Ограничиваем максимум
            'requests': min(features_dict.get('requests', 0) / 10.0, 1.0),
            'replies': min(features_dict.get('replies', 0) / 10.0, 1.0),
            'packet_rate': min(features_dict.get('packet_rate', 0) / 10.0, 1.0),
            'multiple_macs': features_dict.get('multiple_macs', 0),  # Уже 0 или 1
            'request_reply_ratio': min(features_dict.get('request_reply_ratio', 0) / 10.0, 1.0)
        }
        
        # Создаем DataFrame и обрабатываем
        features = pd.DataFrame([model_features])
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Без стандартизации, так как мы уже нормализовали значения
        return features.values
    
    def detect_attack_autoencoder(self, features_dict):
        """Обнаружение аномального поведения с помощью автоэнкодера"""
        try:
            # Предобработка признаков для модели
            features_scaled = self.preprocess_features(features_dict)
            
            # Получение реконструкции от автоэнкодера
            reconstructions = self.model.predict(features_scaled, verbose=0)
            
            # Вычисление ошибки реконструкции (MAE)
            reconstruction_error = np.mean(np.abs(reconstructions - features_scaled), axis=1)[0]
            
            # Проверка поведения устройства
            src_mac = features_dict.get('src_mac', '')
            is_suspicious, reason = self.is_suspicious_behavior(
                src_mac, features_dict, reconstruction_error)
            
            # Определение типа возможной атаки на основе признаков
            attack_type = "Неизвестная аномалия"
            
            if is_suspicious:
                if features_dict['packet_rate'] > 5 and features_dict['duplicates'] > 3:
                    attack_type = "ARP flooding"
                elif features_dict['ip_with_multi_mac'] > 0:
                    attack_type = "ARP spoofing (IP с несколькими MAC)"
                elif features_dict['mac_with_multi_ip'] > 0:
                    attack_type = "ARP poisoning (MAC с несколькими IP)"
            
            confidence = min(reconstruction_error / (2 * self.threshold), 0.99) if is_suspicious else 0.0
            
            return is_suspicious, confidence, attack_type, reconstruction_error, reason
            
        except Exception as e:
            print(f"[-] Ошибка при обнаружении атаки через автоэнкодер: {str(e)}")
            return False, 0.0, "Ошибка обнаружения", 0.0, str(e)
    
    def log_packet(self, packet_data, features_dict=None, reconstruction_error=None, reason=None):
        """Логирование информации о пакете"""
        timestamp = datetime.fromtimestamp(packet_data['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"\n[{timestamp}] Получен ARP-пакет:\n"
        log_entry += f"    Тип: {'ARP Reply' if packet_data['opcode'] == 2 else 'ARP Request'}\n"
        log_entry += f"    Отправитель MAC: {packet_data['src_mac']}\n"
        log_entry += f"    Отправитель IP: {packet_data['src_ip']}\n"
        log_entry += f"    Получатель MAC: {packet_data['dst_mac']}\n"
        log_entry += f"    Получатель IP: {packet_data['dst_ip']}\n"
        log_entry += f"    Broadcast: {'Да' if packet_data['is_broadcast'] else 'Нет'}\n"
        
        if features_dict and reconstruction_error is not None:
            log_entry += f"\nСтатистика окна ({self.window_size} сек):\n"
            log_entry += f"    Пакетов в окне: {len(self.packet_buffer)}\n"
            log_entry += f"    Запросов: {features_dict['requests']}\n"
            log_entry += f"    Ответов: {features_dict['replies']}\n"
            log_entry += f"    Дубликатов: {features_dict['duplicates']}\n"
            log_entry += f"    Частота пакетов: {features_dict['packet_rate']:.2f} пак/сек\n"
            log_entry += f"    Ошибка реконструкции: {reconstruction_error:.6f}\n"
            
            # Информация о профиле устройства
            src_mac = features_dict.get('src_mac', '')
            if src_mac in self.device_profiles:
                profile = self.device_profiles[src_mac]
                log_entry += f"    Профиль устройства {src_mac}:\n"
                log_entry += f"        Средняя ошибка: {profile['avg_error']:.6f}\n"
                log_entry += f"        Порог устройства: {profile['threshold']:.6f}\n"
                time_known = time.time() - self.device_first_seen[src_mac]
                log_entry += f"        Известно: {int(time_known)} сек\n"
            
            if reason:
                log_entry += f"    Статус: {reason}\n"
        
        log_entry += "-"*50 + "\n"
        
        # Запись в файл сессии
        with open(self.session_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
    def log_alert(self, confidence, features_dict, attack_type, reconstruction_error, reason):
        """Логирование обнаруженной атаки"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        alert_msg = f"\n[{timestamp}] Обнаружена подозрительная активность!\n"
        alert_msg += f"    Устройство: MAC={features_dict.get('src_mac', '')}, IP={features_dict.get('src_ip', '')}\n"
        alert_msg += f"    Ошибка реконструкции: {reconstruction_error:.6f} (порог: {self.threshold:.6f})\n"
        alert_msg += f"    Уверенность: {confidence:.4f}\n"
        alert_msg += f"    Тип активности: {attack_type}\n"
        alert_msg += f"    Причина: {reason}\n"
        alert_msg += "\nПризнаки:\n"
        
        # Добавляем только числовые признаки
        for key, value in features_dict.items():
            if key not in ['src_mac', 'src_ip']:
                alert_msg += f"    - {key}: {value}\n"
            
        alert_msg += "-"*50 + "\n"
        
        # Запись в файл алертов
        with open(self.alert_file, 'a', encoding='utf-8') as f:
            f.write(alert_msg)
            
        # Вывод в консоль
        print(f"\n[!] {timestamp} - Подозрительная активность! Ошибка: {reconstruction_error:.6f}")
        print(f"[!] Устройство: {features_dict.get('src_mac', '')}")
        print(f"[!] Тип: {attack_type}")
        print(f"[!] Причина: {reason}")
        print(f"[!] Детали записаны в {self.alert_file}\n")
    
    def packet_handler(self, packet):
        """Обработка пакетов"""
        if ARP in packet:
            arp = packet[ARP]
            
            # Извлекаем время и основные признаки
            packet_time = time.time()
            is_broadcast = packet[Ether].dst == "ff:ff:ff:ff:ff:ff"
            src_mac = packet[Ether].src
            
            # Проверяем, не находится ли отправитель в белом списке
            if src_mac in self.whitelist:
                print(f"\n[i] Пропущен пакет от известного устройства: {self.whitelist[src_mac]}")
                return
            
            # Извлекаем данные из пакета
            packet_data = {
                'timestamp': packet_time,
                'src_mac': src_mac,
                'dst_mac': packet[Ether].dst,
                'src_ip': arp.psrc,
                'dst_ip': arp.pdst,
                'opcode': arp.op,
                'is_broadcast': is_broadcast
            }
            
            # Добавляем пакет в буфер
            self.packet_buffer.append(packet_data)
            
            # Удаляем старые пакеты из буфера
            current_time = time.time()
            self.packet_buffer = deque([p for p in self.packet_buffer 
                                     if current_time - p['timestamp'] <= self.window_size])
            
            # Получаем текущее окно пакетов
            window_packets = list(self.packet_buffer)
            
            # Если накопилось достаточно пакетов, анализируем их
            if len(window_packets) >= 3:  # Минимальный размер для анализа
                # Вычисляем признаки
                features_dict = self.calculate_window_features(window_packets)
                
                if features_dict:
                    # Обнаружение атаки с помощью автоэнкодера и поведенческого анализа
                    is_suspicious, confidence, attack_type, reconstruction_error, reason = self.detect_attack_autoencoder(features_dict)
                    
                    # Логируем пакет и статистику
                    self.log_packet(packet_data, features_dict, reconstruction_error, reason)
                    
                    # Логирование при обнаружении подозрительной активности
                    if is_suspicious:
                        self.log_alert(confidence, features_dict, attack_type, reconstruction_error, reason)
                    
                    # Определяем статус устройства
                    if src_mac in self.device_first_seen:
                        time_known = current_time - self.device_first_seen[src_mac]
                        status = f"Период обучения ({int(time_known)}/{self.learning_period} сек)" if time_known < self.learning_period else "Обучение завершено"
                    else:
                        status = "Новое устройство"
                        
                    # Вывод статистики в консоль
                    print(f"\n[i] Пакет от {src_mac} ({status}):")
                    print(f"    Запросов: {features_dict['requests']}")
                    print(f"    Ответов: {features_dict['replies']}")
                    print(f"    Частота пакетов: {features_dict['packet_rate']:.2f} пак/сек")
                    print(f"    Ошибка реконструкции: {reconstruction_error:.6f}")
                    
                    # Если устройство прошло период обучения, показываем его профиль
                    if src_mac in self.device_profiles and time_known >= self.learning_period:
                        profile = self.device_profiles[src_mac]
                        print(f"    Средняя ошибка: {profile['avg_error']:.6f}")
                        print(f"    Порог устройства: {profile['threshold']:.6f}")
                        print(f"    Статус: {'Подозрительно' if is_suspicious else 'Нормально'}")
                    
                    print("-"*50)
    
    def start_sniffing(self):
        """Запуск сниффера для мониторинга ARP трафика"""
        try:
            print("\n" + "="*50)
            print(" Запуск системы обнаружения ARP-атак с поведенческим анализом")
            print("="*50)
            
            if self.interface:
                print(f"[+] Сканирование интерфейса: {self.interface}")
                sniff(iface=self.interface, filter="arp", prn=self.packet_handler, store=0)
            else:
                print(f"[+] Сканирование всех доступных сетевых интерфейсов")
                print(f"[i] Доступные интерфейсы: {', '.join(get_if_list())}")
                sniff(filter="arp", prn=self.packet_handler, store=0)
        except KeyboardInterrupt:
            print("\n[+] Сканирование остановлено пользователем")
        except Exception as e:
            print(f"\n[-] Ошибка при сканировании: {str(e)}")   

def main():
    print("="*80)
    print("             СИСТЕМА ОБНАРУЖЕНИЯ ARP-SPOOFING С ПОВЕДЕНЧЕСКИМ АНАЛИЗОМ")
    print("="*80)
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='ARP Sniffer с поведенческим анализом')
    parser.add_argument('-i', '--interface', help='Сетевой интерфейс для сканирования')
    parser.add_argument('-t', '--threshold', type=float, default=0.17, help='Порог ошибки реконструкции (по умолчанию 0.17)')
    parser.add_argument('-w', '--window', type=int, default=10, help='Размер временного окна в секундах (по умолчанию 10)')
    parser.add_argument('-l', '--learning', type=int, default=30, help='Период обучения для новых устройств в секундах (по умолчанию 30)')
    
    args = parser.parse_args()
    
    # Создание и запуск сниффера
    try:
        sniffer = BehavioralARPSniffer(
            interface=args.interface, 
            window_size=args.window,
            threshold=args.threshold,
            learning_period=args.learning
        )
        sniffer.start_sniffing()
    except Exception as e:
        print(f"[-] Ошибка при создании сниффера: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()