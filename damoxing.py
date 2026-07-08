# -*- coding: utf-8 -*-
import sys
import os
import requests
from datetime import datetime

source_code_path = '/home/sancog/Desktop/longxin/ultralytics'
model_path = '/home/sancog/Desktop/longxin/best.pt'
image_path = '/home/sancog/Desktop/longxin/1372422923_jpg.rf.e09c80241064db7ebea034036a3d24a8.jpg'

API_KEY = "sk-de16ee3ed8834045ba64f86e10e9c71b"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

if source_code_path not in sys.path:
    sys.path.insert(0, source_code_path)

def ask_deepseek(food_name, now_time, health_state):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = (
        f"Time:{now_time}, Food:{food_name}, User health:{health_state}. "
        "Reply in Chinese. "
        "Based on Chinese Dietary Guidelines. "
        "Include: 1. Calorie & nutrition per 100g; 2. Suitable people; 3. Pros and cons; 4. For diabetes, hypertension, fat loss, allergy; 5. Score 0-10; 6. Taboos; 7. Best food collocation."
    )

    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        res = requests.post(DEEPSEEK_URL, headers=headers, json=data)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return "Request error: " + str(e)

try:
    print("Loading model...")
    from ultralytics import YOLO
    os.environ['ULTRALYTICS_PANDAS_DISABLED'] = 'True'

    model = YOLO(model_path)
    results = model.predict(source=image_path, device='cpu', save=False, conf=0.25)

    obj_list = []
    for res in results:
        for cls_id in res.boxes.cls:
            obj_list.append(model.names[int(cls_id)])

    if not obj_list:
        print("No object detected")
        sys.exit()

    target_food = obj_list[0]
    now_t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("================================================")
    print("Detect success!")
    print("Food: " + target_food)
    print("Time: " + now_t)
    print("================================================")

    print("\nPlease select your health status:")
    print("1 - Normal")
    print("2 - Diabetes")
    print("3 - Hypertension")
    print("4 - Fat loss")
    print("5 - Allergy easily")

    choice = input("\nEnter number (1-5): ")

    health_map = {
        "1": "normal",
        "2": "diabetes",
        "3": "hypertension",
        "4": "fat_loss",
        "5": "allergy"
    }

    if choice not in health_map:
        print("Invalid input")
        sys.exit()

    user_status = health_map[choice]
    print("\nYour choice: " + user_status)
    print("Generating diet analysis...\n")

    ans = ask_deepseek(target_food, now_t, user_status)

    print("================================================")
    print("Diet Analysis Result")
    print("================================================")
    print(ans)
    print("================================================")

except Exception as e:
    print("Error: " + str(e))
    import traceback
    traceback.print_exc()