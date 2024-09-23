import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras import models, utils
from pyautogui import click
from PIL import ImageGrab
import pygetwindow as gw
from pynput import mouse
from time import sleep
import numpy as np

window_title = "Roblox"
check = True
enabled = False

def on_click(x, y, button, pressed):
    global enabled
    if str(button) == 'Button.x1' and pressed:
        enabled = not enabled
        if enabled:
            print("Script enabled")
        else:
            print("Script disabled")

mouse_listener = mouse.Listener(on_click=on_click)
mouse_listener.start()

def predict_image(model) -> None:
    global check
    window = gw.getWindowsWithTitle(window_title)
    if window:
        win = window[0]
        left, top, right, bottom = win.left, win.top, win.right, win.bottom
        img = ImageGrab.grab(bbox=((left + (right - left) / 2) - 250, (top + (bottom - top) / 2) - 250, (left + (right - left) / 2) + 250, (top + (bottom - top) / 2) + 250))

        img_array = utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255
        prediction = model.predict(img_array, verbose=0)
        # print(f"{prediction[0][0]:.2f}")
        if prediction > 0.5:
            if check:
                print("Red body detected!")
                click()
            check = False
        else:
            check = True
    else:
        print('Roblox not found.')

model = models.load_model('src/model/blade_ball.keras')

while True:
    if enabled:
        predict_image(model)
    sleep(1e-3)