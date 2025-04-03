import argparse
import os
import time
import numpy as np
import pandas as pd
from scapy.all import *
from scapy.layers.l2 import ARP, Ether
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import threading
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime
from scapy.all import sniff, get_if_list
warnings.filterwarnings('ignore')

class ARPSniffer:
    def __init__(self, model_path, window_size=10, alert_threshold=0.7, interface=None):
        """
        Инициализация сниффера ARP-пакетов
        
        Параметры:
        -----------
        model_path (str): Путь к предварительно обученной модели LSTM
        window_size (int): Размер временного окна в секундах для анализа
        alert_threshold (float): Порог вероятности для срабатывания тревоги (0.0-1.0)
        interface (str): Сетевой интерфейс для сканирования
        """
        print("[+] Инициализация сниффера ARP-атак...")
        self.model_path = model_path
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.interface = interface
        
        # Проверка существования модели
        if not os.path.exists(model_path):
            print(f"[-] Ошибка: Модель {model_path} не найдена")
            # Попытка найти модель в подкаталоге models
            alternative_path = os.path.join("models", "lstm_model.h5")
            if os.path.exists(alternative_path):
                self.model_path = alternative_path
                print(f"[+] Найдена модель в подкаталоге: {alternative_path}")
            else:
                # Полный путь к модели
                full_path = "D:/Проекты/Дипломаня работа/DoFitN/Code/DoFitN/new_code/Test/AE2/models/lstm_model.h5"
                if os.path.exists(full_path):
                    self.model_path = full_path
                    print(f"[+] Найдена модель по альтернативному пути: {full_path}")
                else:
                    print(f"[i] Доступные модели:")
                    for root, dirs, files in os.walk("."):
                        for file in files:
                            if file.endswith(".h5") or file.endswith(".keras"):
                                print(f"    - {os.path.join(root, file)}")
                    sys.exit(1)
            
        # Инициализация буфера пакетов и отслеживания IP-MAC
        self.packet_buffer = deque(maxlen=1000)  # Используем deque с ограничением размера
        self.mac_ip_mappings = {}
        self.last_ip_change = {}  # Время последнего изменения IP-MAC
        self.known_devices = {}  # Словарь известных устройств
        
        # Белый список известных устройств (маршрутизаторы и т.д.)
        self.whitelist = {
            # формат: 'MAC': 'описание'
            '00:ad:24:bf:9d:52': 'Маршрутизатор 192.168.1.1',
            # Здесь можно добавить другие известные устройства
        }
        
        # Загрузка модели
        self.model = None
        self.load_model()
        
        # Создание скейлера для нормализации данных
        self.scaler = StandardScaler()
        
        # Создание директории для логов
        self.logs_dir = "logs"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        
        # Файл для логирования
        self.log_file = os.path.join(self.logs_dir, f"arp_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        print("[+] Сниффер инициализирован и готов к запуску")
    
    def load_model(self):
        """Загрузка предварительно обученной модели"""
        try:
            self.model = load_model(self.model_path)
            print(f"[+] Модель загружена из {self.model_path}")
        except Exception as e:
            print(f"[-] Ошибка при загрузке модели: {str(e)}")
            sys.exit(1)
    
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
        ip_mac_mappings = {}
        changed_mappings = 0
        
        # Считаем количество различных типов пакетов
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
            
            # Считаем запросы и ответы ARP
            if packet['opcode'] == 1:  # request
                requests += 1
            elif packet['opcode'] == 2:  # reply
                replies += 1
                
            # Считаем broadcast и unicast пакеты
            if packet['is_broadcast']:
                broadcast_count += 1
            else:
                unicast_count += 1
                
            # Отслеживаем соответствия IP-MAC для выявления изменений
            if packet['opcode'] == 2:  # Только для ARP ответов
                src_ip = packet['src_ip']
                src_mac = packet['src_mac']
                
                # Проверяем, есть ли уже запись для этого IP
                if src_ip in ip_mac_mappings:
                    if ip_mac_mappings[src_ip] != src_mac:
                        # Проверяем, не является ли это легитимным DHCP обновлением
                        # Повторяем проверку несколько раз за окно времени
                        if time.time() - self.last_ip_change.get(src_ip, 0) > 300:  # 5 минут
                            changed_mappings += 1
                        self.last_ip_change[src_ip] = time.time()
                
                ip_mac_mappings[src_ip] = src_mac
        
        # Расчет статистики
        multiple_macs = len(set(p['src_mac'] for p in window_packets)) > 1
        request_reply_ratio = requests / replies if replies > 0 else requests
        packet_rate = total_packets / time_span if time_span > 0 else 0
        is_broadcast_pct = broadcast_count / total_packets if total_packets > 0 else 0
        
        # Среднее время между пакетами
        avg_time_between_packets = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        
        # Максимальное количество пакетов в секунду
        max_packets_per_second = max(1 / diff if diff > 0 else 0 for diff in time_diffs) if time_diffs else 0
        
        # Формируем словарь признаков
        features_dict = {
            'duplicates': duplicates,
            'requests': requests,
            'replies': replies,
            'packet_rate': packet_rate,
            'multiple_macs': int(multiple_macs),
            'request_reply_ratio': request_reply_ratio,
            'changed_mappings': changed_mappings,
            'is_broadcast_pct': is_broadcast_pct,
            'broadcast_count': broadcast_count,
            'unicast_count': unicast_count,
            'avg_time_between_packets': avg_time_between_packets,
            'max_packets_per_second': max_packets_per_second
        }
        
        return features_dict
    
    def preprocess_features(self, features_dict):
        """Преобразование словаря признаков в формат для модели"""
        # Добавление хешей категориальных признаков
        features = pd.DataFrame([features_dict])
        
        # Замена inf и NaN значений
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Нормализация данных
        if not hasattr(self.scaler, 'mean_'):
            # При первом запуске инициализируем скейлер базовыми параметрами
            self.scaler.mean_ = np.zeros(features.shape[1])
            self.scaler.scale_ = np.ones(features.shape[1])
        
        features_scaled = self.scaler.transform(features)
        
        # Преобразование в 3D-формат для LSTM [samples, time_steps, features]
        features_lstm = features_scaled.reshape(1, 1, features_scaled.shape[1])
        
        return features_lstm
    
    def rule_based_detection(self, features_dict):
        """Обнаружение атаки на основе правил (резервный метод)"""
        is_attack = False
        confidence = 0.0
        
        # Признаки дублирования пакетов - повышаем порог
        if features_dict['duplicates'] > 10:  # Было 5, стало 10
            confidence += 0.2  # Снижаем влияние дубликатов (было 0.3)
        
        # Множественные MAC для одного IP
        if features_dict['multiple_macs'] > 1:  # Было 0, стало 1
            confidence += 0.4  # Снижаем вес (было 0.5)
        
        # Изменение MAC-IP связок - добавляем верификацию
        if features_dict['changed_mappings'] > 2:  # Было 0, стало 2
            confidence += 0.3  # Снижаем вес (было 0.4)
        
        # Высокая частота пакетов
        if features_dict['packet_rate'] > 5.0:  # Было 3.0, стало 5.0
            confidence += 0.2
        
        # Атака обнаруживается только при достаточном уровне уверенности
        if confidence > 0.5:  # Устанавливаем минимальный порог уверенности
            is_attack = True
        
        # Нормализуем уверенность до 0.99
        confidence = min(confidence, 0.99)
        
        return is_attack, confidence
        
    def detect_attack(self, features_lstm):
        """Обнаружение атаки с помощью LSTM модели"""
        try:
            prediction = self.model.predict(features_lstm, verbose=0)[0][0]
            is_attack = prediction >= self.alert_threshold
            return is_attack, prediction
        except Exception as e:
            print(f"[-] Ошибка при обнаружении атаки через LSTM: {str(e)}")
            print("[!] Переключение на обнаружение на основе правил")
            # Извлекаем исходные признаки из тензора
            features_dict = {}
            features_array = features_lstm[0, 0]
            features_names = ['duplicates', 'requests', 'replies', 'packet_rate', 
                             'multiple_macs', 'request_reply_ratio', 'changed_mappings', 
                             'is_broadcast_pct', 'broadcast_count', 'unicast_count', 
                             'avg_time_between_packets', 'max_packets_per_second']
            
            for i, name in enumerate(features_names):
                if i < len(features_array):
                    features_dict[name] = features_array[i]
                else:
                    features_dict[name] = 0
                    
            return self.rule_based_detection(features_dict)
    
    def log_alert(self, prediction, features_dict):
        """Логирование обнаруженной атаки"""
        # Создаем директорию для логов, если не существует
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        log_file = os.path.join('logs', f'arp_attacks_{datetime.now().strftime("%Y%m%d")}.log')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Формируем подробное сообщение
        alert_msg = f"[{timestamp}] Возможная ARP-атака обнаружена! Уверенность: {prediction:.4f}\n"
        alert_msg += "Признаки:\n"
        for key, value in features_dict.items():
            alert_msg += f"  - {key}: {value}\n"
            
        # Дополнительные подсказки о типе атаки
        attack_type = "Неизвестный тип атаки"
        if features_dict['changed_mappings'] > 0:
            attack_type = "ARP spoofing (изменение MAC-IP привязок)"
        elif features_dict['multiple_macs'] > 0:
            attack_type = "ARP poisoning (множественные MAC-адреса)"
        elif features_dict['duplicates'] > 5:
            attack_type = "ARP flooding (большое количество дубликатов)"
        
        alert_msg += f"Предполагаемый тип атаки: {attack_type}\n"
        alert_msg += "-" * 50 + "\n"
        
        # Запись в лог-файл
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(alert_msg)
            
        # Вывод в консоль
        print(f"\n[!] {timestamp} - Обнаружена ARP-атака! Уверенность: {prediction:.4f}")
        print(f"[!] Тип атаки: {attack_type}")
        print(f"[!] Детали записаны в {log_file}\n")
    
    def packet_handler(self, packet):
        """Обработка пакетов"""
        # Проверяем, что это ARP пакет
        if ARP in packet:
            arp = packet[ARP]
            
            # Извлекаем время и основные признаки
            packet_time = time.time()
            is_broadcast = packet[Ether].dst == "ff:ff:ff:ff:ff:ff"
            
            # Вывод информации о пакете
            print("\n" + "="*50)
            print(f"[+] Получен ARP-пакет ({datetime.fromtimestamp(packet_time).strftime('%Y-%m-%d %H:%M:%S')})")
            print(f"    Тип: {'ARP Reply' if arp.op == 2 else 'ARP Request'}")
            print(f"    Отправитель MAC: {packet[Ether].src}")
            print(f"    Отправитель IP: {arp.psrc}")
            print(f"    Получатель MAC: {packet[Ether].dst}")
            print(f"    Получатель IP: {arp.pdst}")
            print(f"    Broadcast: {'Да' if is_broadcast else 'Нет'}")
            print("="*50)
            
            # Извлекаем данные из пакета
            packet_data = {
                'timestamp': packet_time,
                'src_mac': packet[Ether].src,
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
            
            # Вычисляем признаки и проверяем на атаку
            features_dict = self.calculate_window_features(window_packets)
            
            if features_dict:
                # Предобработка признаков для модели
                features_lstm = self.preprocess_features(features_dict)
                
                # Обнаружение атаки
                is_attack, prediction = self.detect_attack(features_lstm)
                
                # Логирование при обнаружении атаки
                if is_attack:
                    self.log_alert(prediction, features_dict)
                    
                # Вывод статистики окна
                print(f"\n[i] Статистика окна ({self.window_size} сек):")
                print(f"    Пакетов в окне: {len(window_packets)}")
                print(f"    Запросов: {features_dict['requests']}")
                print(f"    Ответов: {features_dict['replies']}")
                print(f"    Дубликатов: {features_dict['duplicates']}")
                print(f"    Частота пакетов: {features_dict['packet_rate']:.2f} пак/сек")
                print(f"    Изменений MAC-IP: {features_dict['changed_mappings']}")
                print(f"    Множественные MAC: {'Да' if features_dict['multiple_macs'] else 'Нет'}")
                print(f"    Вероятность атаки: {prediction*100:.2f}%")
                print("-"*50)
    
    def start_sniffing(self):
        """Запуск сниффера для мониторинга ARP трафика"""
        try:
            print("\n" + "="*50)
            print(" Запуск системы обнаружения ARP-атак на основе LSTM")
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
    print("                    СИСТЕМА ОБНАРУЖЕНИЯ ARP-SPOOFING АТАК")
    print("="*80)
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='ARP Sniffer для обнаружения атак с использованием LSTM')
    parser.add_argument('-i', '--interface', help='Сетевой интерфейс для сканирования')
    parser.add_argument('-m', '--model', default='model.h5', help='Путь к модели LSTM (.h5 файл)')
    parser.add_argument('-w', '--window', type=int, default=10, help='Размер временного окна в секундах (по умолчанию 10)')
    parser.add_argument('-t', '--threshold', type=float, default=0.7, help='Порог обнаружения атаки (0-1, по умолчанию 0.7)')
    
    args = parser.parse_args()
    
    # Проверка наличия модели
    if not os.path.exists(args.model):
        print(f"[-] Ошибка: Модель не найдена по пути: {args.model}")
        print(f"[-] Текущая рабочая директория: {os.getcwd()}")
        
        # Попытка найти модель в альтернативных местах
        alt_paths = [
            "models/lstm_model.keras",
            "models/lstm_model.h5",
            "../models/lstm_model.keras", 
            "../models/lstm_model.h5",
            "D:/Проекты/Дипломаня работа/DoFitN/Code/DoFitN/new_code/Test/AE2/models/lstm_model.keras",
            "D:/Проекты/Дипломаня работа/DoFitN/Code/DoFitN/new_code/Test/AE2/models/lstm_model.h5"
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                print(f"[+] Найдена модель по альтернативному пути: {path}")
                args.model = path
                break
        
        if not os.path.exists(args.model):
            print("[-] Не удалось найти модель. Пожалуйста, укажите правильный путь:")
            print("    python sniff.py --model ПУТЬ_К_МОДЕЛИ")
            exit(1)
    
    # Создание и запуск сниффера
    try:
        sniffer = ARPSniffer(
            model_path=args.model,
            interface=args.interface, 
            window_size=args.window,
            alert_threshold=args.threshold
        )
        sniffer.start_sniffing()
    except Exception as e:
        print(f"[-] Ошибка при создании сниффера: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()