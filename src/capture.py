import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras import models,utils
from PIL import ImageGrab
import pygetwindow as gw
from pynput import mouse
import numpy as np
import datetime
import time

window_title = "Roblox"
window = gw.getWindowsWithTitle(window_title)

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

model = models.load_model('src/model/blade_ball.keras')
while True:
    if (enabled and window):
        win = window[0]
        left, top, right, bottom = win.left, win.top, win.right, win.bottom
        img = ImageGrab.grab(bbox=((left + (right - left) / 2) - 250, (top + (bottom - top) / 2) - 250, (left + (right - left) / 2) + 250, (top + (bottom - top) / 2) + 250))
        # img = remove(img)
        img_array = utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
    
        prediction: str = model.predict(img_array, verbose=0)
        # print(f"{int(prediction[0][0]):.2f}")
        # print(f"{prediction[0][0]:.2f}")
        date = datetime.datetime.now()
        hr = date.strftime("%H")
        min = date.strftime("%M")
        sec = date.strftime("%S")
        ms = date.strftime("%f")
        if prediction > 0.1:
            img.save(f'saves/red_body/{hr}_{min}_{sec}_{ms}.png')
        else:
            img.save(f'saves/non_red_body/{hr}_{min}_{sec}_{ms}.png')
    time.sleep(0.01)