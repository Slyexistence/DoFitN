from scapy.all import *
import signal
import sys

def signal_handler(sig, frame):
    print("\nARP Прослушка отключена пользователем.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def process_arp_packet(packet):
    if packet.haslayer(ARP):
        arp = packet[ARP]
        
        # Определяем тип ARP-пакета
        if arp.op == 1:
            operation = "Запрос"
        elif arp.op == 2:
            operation = "Ответ"
        else:
            operation = "Неизвестная операция"
        
        # Извлекаем информацию из ARP-пакета
        sender_mac = arp.hwsrc
        sender_ip = arp.psrc
        target_mac = arp.hwdst
        target_ip = arp.pdst
        
        # Форматируем вывод для консоли
        print(f"[ARP {operation}]")
        print(f"Sender: {sender_mac} ({sender_ip})")
        print(f"Target: {target_mac} ({target_ip})")
        print("-" * 40)

if __name__ == "__main__":
    print("ARP Прослушка включена. Горячие клавиши Ctrl+C для остановки...\n")
    sniff(filter="arp", prn=process_arp_packet, store=0)