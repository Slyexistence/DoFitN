import os
import math
import numpy as np
import pandas as pd
from scapy.all import rdpcap, ARP, Ether
from collections import defaultdict, Counter
import time
import ipaddress
from scipy.stats import entropy

def process_pcap_files(pcap_directory, output_csv, is_anomaly=False):
    """
    Обрабатывает все pcap/pcapng файлы в директории, извлекает признаки и сохраняет их в CSV.
    
    Параметры:
    pcap_directory (str): Путь к директории с pcap файлами
    output_csv (str): Путь для сохранения результатов в CSV
    is_anomaly (bool): Флаг, указывающий являются ли данные аномальными (для метки класса)
    """
    all_features = []
    
    # Получаем список всех pcap/pcapng файлов в директории
    pcap_files = [f for f in os.listdir(pcap_directory) if f.endswith(('.pcap', '.pcapng'))]
    
    print(f"Найдено {len(pcap_files)} pcap файлов в директории {pcap_directory}")
    
    for pcap_file in pcap_files:
        print(f"Обработка файла: {pcap_file}")
        file_path = os.path.join(pcap_directory, pcap_file)
        
        # Извлечь признаки из файла
        file_features = extract_features_from_pcap(file_path, is_anomaly, pcap_file)
        all_features.extend(file_features)
    
    # Создаем DataFrame и сохраняем в CSV
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_csv, index=False)
    print(f"Признаки сохранены в {output_csv}")
    
    return features_df

def extract_features_from_pcap(pcap_file, is_anomaly, session_id):
    """
    Извлекает признаки из pcap файла для обнаружения ARP-спуфинга.
    
    Параметры:
    pcap_file (str): Путь к pcap файлу
    is_anomaly (bool): Является ли данный файл примером аномалии
    session_id (str): Идентификатор сессии (имя файла)
    
    Возвращает:
    list: Список словарей с признаками для каждого ARP-пакета
    """
    packets = rdpcap(pcap_file)
    
    # Словари для хранения информации о ARP-пакетах
    arp_packets = []
    mac_ip_pairs = defaultdict(set)  # MAC -> set of IP addresses
    ip_mac_pairs = defaultdict(set)  # IP -> set of MAC addresses
    mac_timestamps = defaultdict(list)  # MAC -> list of timestamps
    request_reply_pairs = {}  # (src_mac, target_ip) -> timestamp для отслеживания запросов
    
    # Первый проход - сбор базовой информации
    for packet in packets:
        if ARP in packet:
            arp = packet[ARP]
            timestamp = float(packet.time)
            
            # Извлекаем основную информацию
            src_mac = arp.hwsrc
            dst_mac = arp.hwdst
            src_ip = arp.psrc
            dst_ip = arp.pdst
            opcode = arp.op  # 1=request, 2=reply
            
            # Собираем ARP-пакеты для дальнейшей обработки
            arp_packets.append({
                'timestamp': timestamp,
                'src_mac': src_mac,
                'dst_mac': dst_mac,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'opcode': opcode,
                'packet': packet
            })
            
            # Обновляем соответствия MAC-IP
            if src_ip != '0.0.0.0' and src_mac != '00:00:00:00:00:00':
                mac_ip_pairs[src_mac].add(src_ip)
                ip_mac_pairs[src_ip].add(src_mac)
            
            # Сохраняем временные метки для каждого MAC
            mac_timestamps[src_mac].append(timestamp)
            
            # Отслеживаем запросы и ответы для измерения времени ответа
            if opcode == 1:  # request
                key = (src_mac, dst_ip)
                if key not in request_reply_pairs:
                    request_reply_pairs[key] = timestamp
            elif opcode == 2:  # reply
                key = (dst_mac, src_ip)  # Инвертированная пара для соответствия запросу
                # Обновляем при нахождении более раннего запроса
                request_reply_pairs[key] = request_reply_pairs.get(key, timestamp)
    
    # Сортируем пакеты по времени
    arp_packets.sort(key=lambda x: x['timestamp'])
    
    # Вычисляем признаки для каждого пакета
    all_features = []
    
    # Словари для отслеживания состояния и частоты
    packet_counts = defaultdict(int)  # Счетчик пакетов
    duplicate_counts = defaultdict(int)  # Счетчик дубликатов
    request_counts = defaultdict(int)  # Счетчик запросов
    reply_counts = defaultdict(int)  # Счетчик ответов
    mac_last_seen = {}  # Когда последний раз видели MAC
    
    # Второй проход - вычисление признаков
    for i, packet_info in enumerate(arp_packets):
        src_mac = packet_info['src_mac']
        dst_mac = packet_info['dst_mac']
        src_ip = packet_info['src_ip']
        dst_ip = packet_info['dst_ip']
        opcode = packet_info['opcode']
        timestamp = packet_info['timestamp']
        packet = packet_info['packet']
        
        # 1. Базовые признаки
        features = {
            'timestamp': timestamp,
            'src_mac': src_mac,
            'dst_mac': dst_mac,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'opcode': opcode,
        }
        
        # 2. Признак широковещательности (broadcast)
        is_broadcast = (dst_mac == 'ff:ff:ff:ff:ff:ff')
        features['is_broadcast'] = is_broadcast
        
        # 3. Счетчики и частоты
        packet_counts[(src_mac, dst_ip)] += 1
        
        # Определяем дубликаты (тот же src_mac, src_ip, dst_ip)
        key = (src_mac, src_ip, dst_ip)
        duplicate_counts[key] += 1
        features['duplicates'] = duplicate_counts[key] - 1  # -1 потому что первый пакет не дубликат
        
        # Счетчики запросов и ответов
        if opcode == 1:  # request
            request_counts[(src_mac, dst_ip)] += 1
        else:  # reply
            reply_counts[(src_mac, dst_ip)] += 1
        
        features['requests'] = request_counts[(src_mac, dst_ip)]
        features['replies'] = reply_counts[(src_mac, dst_ip)]
        
        # 4. Частота пакетов (packet rate)
        packet_rate = 0
        if src_mac in mac_last_seen:
            time_diff = timestamp - mac_last_seen[src_mac]
            if time_diff > 0:
                # Количество пакетов в единицу времени
                packet_rate = 1 / time_diff
        features['packet_rate'] = packet_rate
        
        # Обновляем время последнего наблюдения MAC
        mac_last_seen[src_mac] = timestamp
        
        # 5. Признак множественных MAC-адресов для одного IP
        multiple_macs = 1 if len(ip_mac_pairs.get(src_ip, set())) > 1 else 0
        features['multiple_macs'] = multiple_macs
        
        # 6. Соотношение запросов/ответов
        req_count = request_counts[(src_mac, dst_ip)]
        rep_count = reply_counts[(src_mac, dst_ip)]
        if rep_count > 0:
            req_reply_ratio = req_count / rep_count
        else:
            req_reply_ratio = float('inf')  # Избегаем деления на ноль
        features['request_reply_ratio'] = req_reply_ratio
        
        # 7. Новые признаки
        
        # 7.1. Интервал между последовательными пакетами от MAC
        mac_times = mac_timestamps[src_mac]
        if len(mac_times) > 1:
            sorted_times = sorted(mac_times)
            idx = sorted_times.index(timestamp)
            if idx > 0:
                features['time_since_last_packet'] = timestamp - sorted_times[idx-1]
            else:
                features['time_since_last_packet'] = 0
        else:
            features['time_since_last_packet'] = 0
        
        # 7.2. Количество уникальных IP для этого MAC
        features['unique_ip_count'] = len(mac_ip_pairs[src_mac])
        
        # 7.3. Энтропия интервалов (для обнаружения регулярных шаблонов)
        if len(mac_times) > 5:  # Нужно достаточно данных для расчета энтропии
            intervals = [mac_times[i+1] - mac_times[i] for i in range(len(mac_times)-1)]
            # Преобразуем интервалы в дискретные бины для расчета энтропии
            bins = np.histogram(intervals, bins=10)[0]
            if sum(bins) > 0:
                features['interval_entropy'] = entropy(bins+1)  # +1 чтобы избежать log(0)
            else:
                features['interval_entropy'] = 0
        else:
            features['interval_entropy'] = 0
            
        # 7.4. Время ответа (если применимо)
        if opcode == 2:  # Это ответ
            request_key = (dst_mac, src_ip)  # Обратный ключ для поиска запроса
            if request_key in request_reply_pairs:
                request_time = request_reply_pairs[request_key]
                features['response_time'] = timestamp - request_time
            else:
                features['response_time'] = -1  # Ответ без запроса - подозрительно
        else:
            features['response_time'] = 0  # Для запросов не применимо
            
        # 7.5. Признак частоты запросов к разным IP от одного MAC
        # Количество разных целевых IP, к которым обращался этот MAC
        target_ips = sum(1 for k in request_counts if k[0] == src_mac)
        features['target_ip_diversity'] = target_ips
        
        # 7.6. Является ли IP шлюзом (обычно заканчивается на .1)
        try:
            ip_obj = ipaddress.IPv4Address(src_ip)
            if not ip_obj.is_private:  # Публичный IP
                features['is_gateway'] = 0
            else:
                # Эвристика: частые окончания для шлюзов
                last_octet = int(src_ip.split('.')[-1])
                features['is_gateway'] = 1 if last_octet in (1, 254) else 0
        except:
            features['is_gateway'] = 0  # Неправильный формат IP
        
        # 7.7. Количество запросов без ответов и процент ответов
        if req_count > 0:
            features['unanswered_requests'] = max(0, req_count - rep_count)
            features['reply_percentage'] = (rep_count / req_count) * 100
        else:
            features['unanswered_requests'] = 0
            features['reply_percentage'] = 0
        
        # 7.8. Логарифмическая шкала для скорости пакетов
        features['log_packet_rate'] = np.log1p(packet_rate)  # log(1+x) для избежания log(0)
        
        # Добавляем метку класса и идентификатор сессии
        features['label'] = 1 if is_anomaly else 0
        features['session_id'] = session_id
        
        all_features.append(features)
    
    return all_features

# Пример использования:
if __name__ == "__main__":
    # Укажите путь к директории с нормальными pcap файлами
    normal_dir = r"D:\Проекты\Дипломаня работа\DoFitN\Data\only_arp\N"
    # Укажите путь к директории с аномальными pcap файлами
    anomaly_dir = r"D:\Проекты\Дипломаня работа\DoFitN\Data\only_arp\A"
    
    # Обработка нормальных данных
    normal_features = process_pcap_files(normal_dir, "normal_features.csv", is_anomaly=False)
    
    # Обработка аномальных данных
    anomaly_features = process_pcap_files(anomaly_dir, "anomaly_features.csv", is_anomaly=True)
    
    # Объединение данных
    all_features = pd.concat([normal_features, anomaly_features])
    all_features.to_csv("combined_features.csv", index=False)
    
    print("Обработка завершена. Признаки сохранены.")