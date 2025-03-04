from scapy.all import *
import signal
import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from datetime import datetime
import ipaddress

class ARPMonitor:
    def __init__(self, model_dir='models', window_size=10):
        self.window_size = window_size
        self.buffer = []
        self.mac_cache = {}
        self.ip_cache = {}
        self.mac_counter = 0
        self.ip_counter = 0
        self.arp_cache = {}  # Для отслеживания ARP-записей
        self.load_model(model_dir)
        print(f"Инициализация завершена. Размер окна: {window_size}")
        
    def load_model(self, model_dir):
        try:
            model_path = os.path.join(model_dir, 'model.h5')
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            sys.exit(1)

    def normalize_mac(self, mac):
        if mac not in self.mac_cache:
            self.mac_cache[mac] = self.mac_counter
            self.mac_counter += 1
        return self.mac_cache[mac] / max(len(self.mac_cache), 1)

    def normalize_ip(self, ip):
        if ip not in self.ip_cache:
            try:
                ip_int = int(ipaddress.ip_address(ip))
                self.ip_cache[ip] = ip_int
            except:
                self.ip_cache[ip] = self.ip_counter
                self.ip_counter += 1
        return self.ip_cache[ip] / max(len(self.ip_cache), 1)

    def normalize_timestamp(self, timestamp):
        return (timestamp % 86400) / 86400  # Нормализация времени в пределах дня

    def normalize_operation(self, op):
        return op / 2  # ARP operations: 1 (request) or 2 (reply)

    def predict_arp_spoofing(self, packet):
        if not packet.haslayer(ARP):
            return

        arp = packet[ARP]
        timestamp = packet.time
        
        # Вывод информации о каждом ARP-пакете
        print(f"\n[+] Получен ARP-пакет:")
        print(f"    Тип: {'ARP Reply' if arp.op == 2 else 'ARP Request'}")
        print(f"    Отправитель MAC: {arp.hwsrc}")
        print(f"    Отправитель IP: {arp.psrc}")
        print(f"    Получатель MAC: {arp.hwdst}")
        print(f"    Получатель IP: {arp.pdst}")
        
        # Извлечение и нормализация признаков
        features = [
            self.normalize_mac(arp.hwsrc),
            self.normalize_ip(arp.psrc),
            self.normalize_mac(arp.hwdst),
            self.normalize_ip(arp.pdst),
            self.normalize_timestamp(timestamp),
            self.normalize_operation(arp.op)
        ]
        
        self.buffer.append(features)
        print(f"    Размер буфера: {len(self.buffer)}/{self.window_size}")
        
        if len(self.buffer) >= self.window_size:
            sequence = np.array(self.buffer[-self.window_size:])
            sequence = sequence.reshape((1, self.window_size, len(features)))
            
            prediction = self.model.predict(sequence, verbose=0).ravel()[0]
            probability = prediction * 100
            
            print(f"    Результат анализа: {probability:.2f}%")
            
            threshold = 50
            if probability > threshold:
                print(f"\n[!] ВНИМАНИЕ! Обнаружен подозрительный ARP-пакет:")
                print(f"    Отправитель MAC: {arp.hwsrc}")
                print(f"    Отправитель IP: {arp.psrc}")
                print(f"    Получатель MAC: {arp.hwdst}")
                print(f"    Получатель IP: {arp.pdst}")
                print(f"    Вероятность атаки: {probability:.2f}%")
                print(f"    Время: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            self.buffer.pop(0)

def signal_handler(sig, frame):
    print("\nARP Мониторинг остановлен.")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n=== Настройка системы обнаружения ARP-атак ===")
    custom_model_path = r"D:\Проекты\Дипломаня работа\DoFitN\Code\DoFitN\main\model\v1"
    print(f"Путь к модели: {custom_model_path}")
    
    monitor = ARPMonitor(model_dir=custom_model_path)
    print("\nARP Мониторинг запущен. Нажмите Ctrl+C для остановки...")
    print("Ожидание ARP-пакетов...\n")
    
    sniff(filter="arp", prn=monitor.predict_arp_spoofing, store=0)