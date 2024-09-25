import os
import socket
import subprocess
import sys
import time
from context import path
import pyautogui
# import pygetwindow as gw


def listen_for_status():
    i = 100
    exit = False
    while i:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(('localhost', 65432))
                status = s.recv(1024).decode()  # Receive the status
                print("Received status:", status)
                if status == "running":
                    print("Application is now running!")
                    s.send("exit".encode())
                    exit = True
                    # break  # Exit the loop if status is running
            except ConnectionRefusedError:
                print(f"Connection refused. Retrying... {i} Tries Left")
                if exit:
                    return False

            time.sleep(1)  # Wait before trying again
        i -= 1
    return True


def run_and_find_window(script_path):
    failed = 0
    try:
        for i in range(100):
            out_file = "tmp/debug.txt"
            with open(out_file, "w") as output_file:
                app_process = subprocess.Popen(
                    ["python", script_path], stdout=None, stderr=output_file)
                t = listen_for_status()
                if t:
                    app_process.kill()
                app_process.wait()
            with open(out_file, 'r')as f:
                a = f.readlines()
                print(a)
                x = ["invalid command name" in line for line in a]
                print(x)
                if any(x):
                    print("noooo")
                    failed += 1
    except:
        pass
    print(f"{failed}/{i} had an error")


if __name__ == "__main__":
    script_name = 'main.py'
    script_path = os.path.join(path, script_name)
    run_and_find_window(script_path)
