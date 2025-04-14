#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import joblib
import pandas as pd
import numpy as np
from scapy.all import sniff, ARP
from scapy.layers.l2 import Ether
import collections
import socket
import struct

# Глобальные переменные для хранения истории ARP-пакетов
arp_history = collections.defaultdict(list)
sessions = {}
session_features = {}
session_counter = 0

# Путь к обученной модели
MODEL_PATH = r'D:\Проекты\Дипломаня работа\results\mlp_model_best.joblib'

# Загрузка модели
print("Загрузка модели из", MODEL_PATH)
try:
    model = joblib.load(MODEL_PATH)
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    # Пробуем альтернативный путь
    try:
        alt_path = 'results/mlp_model.joblib'
        print(f"Пробуем загрузить модель из {alt_path}")
        model = joblib.load(alt_path)
        print("Модель успешно загружена из альтернативного пути")
    except Exception as e2:
        print(f"Ошибка загрузки модели из альтернативного пути: {e2}")
        exit(1)

def ip2int(addr):
    """Преобразование IP-адреса в целое число"""
    return struct.unpack("!I", socket.inet_aton(addr))[0]

def extract_features_from_packet(packet):
    """Извлечение признаков из пакета ARP"""
    global session_counter, sessions, session_features
    
    # Базовые данные из пакета
    if ARP in packet:
        src_mac = packet[ARP].hwsrc
        dst_mac = packet[ARP].hwdst
        src_ip = packet[ARP].psrc
        dst_ip = packet[ARP].pdst
        op_code = packet[ARP].op  # 1=запрос, 2=ответ
        
        # Создаем сессионный ключ
        session_key = f"{src_ip}_{dst_ip}_{src_mac}_{dst_mac}"
        
        # Если это новая сессия, назначаем ей ID
        if session_key not in sessions:
            sessions[session_key] = session_counter
            session_counter += 1
            session_features[session_key] = {
                "timestamp": time.time(),
                "src_mac": src_mac,
                "dst_mac": dst_mac,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "session_id": sessions[session_key],
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
                "mac_unique_count": 2  # начинаем с src_mac и dst_mac
            }
            arp_history[session_key] = []
        
        # Обновляем количество пакетов и типы операций
        session_features[session_key]["packet_count"] += 1
        if op_code == 1:  # ARP запрос
            session_features[session_key]["request_count"] += 1
        elif op_code == 2:  # ARP ответ
            session_features[session_key]["reply_count"] += 1
        
        # Добавляем текущий пакет в историю
        # Сохраняем только время получения, а не весь пакет
        arp_history[session_key].append(time.time())
        
        # Обновляем признаки интервалов
        if len(arp_history[session_key]) > 1:
            intervals = np.diff(arp_history[session_key])
            session_features[session_key]["interval_mean"] = np.mean(intervals)
            session_features[session_key]["interval_std"] = np.std(intervals) if len(intervals) > 1 else 0
            session_features[session_key]["interval_min"] = np.min(intervals)
            session_features[session_key]["interval_max"] = np.max(intervals)
        
        # Количество уникальных адресов
        # Исправлено: в arp_history хранятся только временные метки, а не пакеты
        # Обновляем данные о уникальных адресах в session_features напрямую
        if src_ip not in session_features[session_key].get("unique_src_ips", set()):
            if "unique_src_ips" not in session_features[session_key]:
                session_features[session_key]["unique_src_ips"] = set()
            session_features[session_key]["unique_src_ips"].add(src_ip)
            
        if dst_ip not in session_features[session_key].get("unique_dst_ips", set()):
            if "unique_dst_ips" not in session_features[session_key]:
                session_features[session_key]["unique_dst_ips"] = set()
            session_features[session_key]["unique_dst_ips"].add(dst_ip)
            
        if src_mac not in session_features[session_key].get("unique_src_macs", set()):
            if "unique_src_macs" not in session_features[session_key]:
                session_features[session_key]["unique_src_macs"] = set()
            session_features[session_key]["unique_src_macs"].add(src_mac)
            
        if dst_mac not in session_features[session_key].get("unique_dst_macs", set()):
            if "unique_dst_macs" not in session_features[session_key]:
                session_features[session_key]["unique_dst_macs"] = set()
            session_features[session_key]["unique_dst_macs"].add(dst_mac)
        
        # Обновляем счетчики уникальных адресов
        session_features[session_key]["src_unique_count"] = len(session_features[session_key].get("unique_src_ips", set()))
        session_features[session_key]["dst_unique_count"] = len(session_features[session_key].get("unique_dst_ips", set()))
        
        all_ips = set()
        all_ips.update(session_features[session_key].get("unique_src_ips", set()))
        all_ips.update(session_features[session_key].get("unique_dst_ips", set()))
        session_features[session_key]["ip_unique_count"] = len(all_ips)
        
        all_macs = set()
        all_macs.update(session_features[session_key].get("unique_src_macs", set()))
        all_macs.update(session_features[session_key].get("unique_dst_macs", set()))
        session_features[session_key]["mac_unique_count"] = len(all_macs)
        
        # Дополнительные признаки
        duration = max(0.001, time.time() - session_features[session_key]["timestamp"])
        session_features[session_key]["arp_rate"] = session_features[session_key]["packet_count"] / duration
        
        # Избегаем деления на ноль
        if session_features[session_key]["reply_count"] > 0:
            session_features[session_key]["request_reply_ratio"] = session_features[session_key]["request_count"] / session_features[session_key]["reply_count"]
        else:
            session_features[session_key]["request_reply_ratio"] = session_features[session_key]["request_count"]  # Если ответов нет, то соотношение равно количеству запросов
        
        # Конвертируем IP в числовой формат для использования в модели
        session_features[session_key]["src_ip_int"] = ip2int(src_ip)
        session_features[session_key]["dst_ip_int"] = ip2int(dst_ip)
        
        return session_features[session_key]
    
    return None

def preprocess_features(features):
    """Предобработка признаков для модели"""
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

def detect_spoofing(packet):
    """Обнаружение ARP-spoofing с использованием модели"""
    if ARP in packet:
        # Извлекаем признаки из пакета
        features = extract_features_from_packet(packet)
        
        # Проверяем, достаточно ли собрано признаков для анализа
        # Для более точного анализа ждем несколько пакетов в сессии
        if features and features["packet_count"] >= 3:
            # Предобрабатываем признаки
            df = preprocess_features(features)
            
            # Прогноз с использованием модели
            try:
                # Исправлено: добавлена проверка типа для предотвращения ошибки индексации
                prediction_result = model.predict(df)
                prediction = prediction_result[0] if hasattr(prediction_result, '__getitem__') else prediction_result
                
                proba_result = model.predict_proba(df)
                # Проверка, является ли proba_result массивом или списком
                if hasattr(proba_result, '__getitem__'):
                    # Если это массив/список, проверяем его первый элемент
                    first_element = proba_result[0]
                    if hasattr(first_element, '__getitem__'):
                        # Если первый элемент тоже массив/список, берем его второй элемент
                        probability = first_element[1]
                    else:
                        # Если первый элемент не массив/список, берем его как вероятность
                        probability = first_element
                else:
                    # Если это не массив/список (например, скаляр), используем значение напрямую
                    probability = proba_result
                
                # Выводим результат
                src_mac = features["src_mac"]
                dst_mac = features["dst_mac"]
                src_ip = features["src_ip"]
                dst_ip = features["dst_ip"]
                
                if prediction == 1:
                    print("\n[ВНИМАНИЕ] Обнаружен возможный ARP-spoofing!")
                    print(f"Вероятность атаки: {probability:.4f}")
                    print(f"Источник: {src_ip} ({src_mac})")
                    print(f"Назначение: {dst_ip} ({dst_mac})")
                    print(f"Количество пакетов: {features['packet_count']}")
                    print(f"Соотношение запросов/ответов: {features['request_reply_ratio']:.2f}")
                    print("=" * 60)
                else:
                    if features["packet_count"] % 10 == 0:  # Показываем информацию о нормальном трафике реже
                        print(f"\n[НОРМАЛЬНО] ARP трафик ({src_ip} -> {dst_ip})")
                        print(f"Вероятность атаки: {probability:.4f}")
                        print("-" * 40)
            
            except Exception as e:
                print(f"Ошибка при анализе пакета: {e}")
                # Подробная информация об ошибке
                import traceback
                traceback.print_exc()
                
                # Вывод доступных признаков для отладки
                print("Доступные признаки:", df.columns.tolist())
                print("Размерность данных:", df.shape)

def packet_callback(packet):
    """Обработчик пакетов"""
    if ARP in packet:
        op_type = "запрос" if packet[ARP].op == 1 else "ответ"
        print(f"ARP {op_type}: {packet[ARP].psrc} ({packet[ARP].hwsrc}) -> {packet[ARP].pdst} ({packet[ARP].hwdst})")
        
        # Анализируем пакет на наличие ARP-spoofing
        detect_spoofing(packet)

def main():
    print("Запуск сниффера ARP-трафика для обнаружения ARP-spoofing атак...")
    print("Используемая модель:", MODEL_PATH)
    print("=" * 60)
    print("Ожидание ARP пакетов...\n")
    
    try:
        # Запускаем сниффер с фильтром только для ARP-пакетов
        sniff(filter="arp", prn=packet_callback, store=0)
    except KeyboardInterrupt:
        print("\nСниффер остановлен пользователем.")
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 