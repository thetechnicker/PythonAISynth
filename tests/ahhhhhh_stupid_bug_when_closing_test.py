import os
import subprocess
import sys
import time
from context import path
import pyautogui
# import pygetwindow as gw


def run_and_find_window(script_path, window_title):

    app_process = subprocess.Popen(["python", script_path])

    time.sleep(2)

    try:
        # window = gw.getWindowsWithTitle(window_title)[0]
        # window.activate()
        # print(f"Found window: {window.title}")

        # window_center = window.center
        # pyautogui.click(window_center.x, window_center.y)

        pyautogui.hotkey('alt', 'f4')

    except IndexError:
        pass
        # print(f"No window found with title: {window_title}")

    app_process.wait()


if __name__ == "__main__":
    script_name = 'main.py'
    script_path = os.path.join(path, script_name)
    window_title = 'Your Window Title'
    run_and_find_window(script_path, window_title)
