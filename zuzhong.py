# -*- coding: utf-8 -*-
import sys
import os

# ==========================================
# 提前设置环境变量，避免 OpenCV 加载 GStreamer 报错
# ==========================================
os.environ["OPENCV_VIDEOIO_PRIORITY_LIST"] = "V4L2"  # 优先 OpenCV 使用 V4L2
os.environ["GST_PLUGIN_SCANNER"] = "0"               # 禁用 GStreamer 插件扫描
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"             # 屏蔽 C++ 底层日志

import requests
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import cv2
from PIL import Image, ImageTk

# ==========================================
# 路径配置 (请根据实际修改)
# ==========================================
source_code_path = '/home/sancog/Desktop/longxin/ultralytics'
model_path = '/home/sancog/Desktop/longxin/best.pt'
logo_path = '0231e820-80a9-4ea7-8c73-eb0fc068a936.png'

API_KEY = "sk-de16ee3ed8834045ba64f86e10e9c71b"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

os.environ['ULTRALYTICS_PANDAS_DISABLED'] = 'True'

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
        "Include: 1. Calorie & nutrition per 100g; 2. Suitable people; 3. Pros and cons; "
        "4. For diabetes, hypertension, fat loss, allergy; 5. Score 0-10; 6. Taboos; 7. Best food collocation."
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
        return "请求失败: " + str(e)


class LongxinDietAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("膳食分析系统 - 龙芯版")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.color_bg = "#FFF5F5"
        self.color_primary = "#A30014"
        self.color_secondary = "#E6001F"
        self.color_card = "#FFFFFF"
        self.color_text = "#2C2222"
        self.color_input_bg = "#FFFBFB"
        
        self.root.configure(bg=self.color_bg, padx=15, pady=15)
        self.model = None 
        self.current_frame = None  
        self.is_analyzing = False
        
        self.font_title = ("Microsoft YaHei", 20, "bold")
        self.font_section = ("Microsoft YaHei", 12, "bold")
        self.font_normal = ("Microsoft YaHei", 11)
        self.font_btn = ("Microsoft YaHei", 13, "bold")
        
        self.setup_styles()
        self.create_layout()
        
        # 初始化摄像头 API 使用 V4L2
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            messagebox.showerror("初始化失败", "摄像头无法打开，请检查设备或权限")
        else:
            self.update_camera_feed()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Card.TLabelframe", background=self.color_card, bordercolor="#FFE0E0", borderwidth=1.5)
        style.configure("Card.TLabelframe.Label", background=self.color_card, foreground=self.color_primary, font=self.font_section)
        style.configure("Action.TButton", font=self.font_normal, background=self.color_secondary, foreground="white")
        style.map("Action.TButton", background=[("active", self.color_primary)])
        # 单选按钮字体放大到14号
        style.configure("TRadiobutton", font=("Microsoft YaHei", 14), background=self.color_card, foreground=self.color_text)
        style.map("TRadiobutton", foreground=[("selected", self.color_primary)])
        style.configure("Exit.TButton", font=("Microsoft YaHei", 12, "bold"), background="#333333", foreground="white")
        style.map("Exit.TButton", background=[("active", "#000000")])

    def create_layout(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=6) 
        self.root.columnconfigure(1, weight=4) 

        self.frame_left = ttk.LabelFrame(self.root, text=" 实时画面 (摄像头识别) ", style="Card.TLabelframe")
        self.frame_left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.lbl_camera = tk.Label(self.frame_left, bg="black")
        self.lbl_camera.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.lbl_camera.bind("<Configure>", self.on_camera_resize)
        self.camera_width = 640
        self.camera_height = 480

        self.frame_right = tk.Frame(self.root, bg=self.color_bg)
        self.frame_right.grid(row=0, column=1, sticky="nsew")

        frame_header = tk.Frame(self.frame_right, bg=self.color_bg)
        frame_header.pack(fill=tk.X, pady=(0, 15))

        if os.path.exists(logo_path):
            try:
                img = Image.open(logo_path)
                scale_height = 45
                scale_width = int((scale_height / float(img.size[1])) * img.size[0])
                img = img.resize((scale_width, scale_height), Image.Resampling.LANCZOS)
                self.logo_image = ImageTk.PhotoImage(img)
                lbl_logo = tk.Label(frame_header, image=self.logo_image, bg=self.color_bg)
                lbl_logo.pack(side=tk.LEFT, padx=(0, 15))
            except Exception as e:
                pass

        lbl_title = tk.Label(frame_header, text="膳食分析仪", font=self.font_title, bg=self.color_bg, fg=self.color_primary)
        lbl_title.pack(side=tk.LEFT, anchor=tk.CENTER)
        
        btn_exit = ttk.Button(frame_header, text="退 出", style="Exit.TButton", command=self.on_closing)
        btn_exit.pack(side=tk.RIGHT, ipady=3, ipadx=5)

        frame_health = ttk.LabelFrame(self.frame_right, text=" 1. 健康状况选择 ", style="Card.TLabelframe", padding=10)
        frame_health.pack(fill=tk.X, pady=(0, 15))

        self.health_var = tk.StringVar(value="normal")
        health_options = [
            ("普通 (Normal)", "normal"),
            ("糖尿病 (Diabetes)", "diabetes"),
            ("高血压 (Hypertension)", "hypertension"),
            ("减脂 (Fat loss)", "fat_loss"),
            ("易过敏 (Allergy)", "allergy")
        ]

        # 放大单选按钮行间距，避免文字拥挤
        for text, val in health_options:
            rb = ttk.Radiobutton(frame_health, text=text, value=val, variable=self.health_var)
            rb.pack(anchor=tk.W, padx=10, pady=5)

        self.btn_analyze = tk.Button(
            self.frame_right, 
            text="开始识别分析", 
            font=self.font_btn, 
            bg=self.color_primary, 
            fg="white", 
            activebackground=self.color_secondary, 
            activeforeground="white", 
            relief=tk.FLAT,
            cursor="hand2",
            command=self.start_analysis_thread, 
            height=2
        )
        self.btn_analyze.pack(fill=tk.X, pady=(0, 15))

        frame_result = ttk.LabelFrame(self.frame_right, text=" 2. 分析结果 ", style="Card.TLabelframe", padding=10)
        frame_result.pack(fill=tk.BOTH, expand=True)

        # 分析结果文本框字体放大到14号
        self.text_result = scrolledtext.ScrolledText(
            frame_result, 
            wrap=tk.WORD, 
            font=("Microsoft YaHei", 14), 
            bg=self.color_input_bg, 
            fg=self.color_text, 
            insertbackground=self.color_primary, 
            relief=tk.SOLID, 
            bd=1
        )
        self.text_result.pack(fill=tk.BOTH, expand=True)

    def on_camera_resize(self, event):
        if event.width > 10 and event.height > 10:
            self.camera_width = event.width
            self.camera_height = event.height

    def update_camera_feed(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                frame_w, frame_h = img.size
                target_w, target_h = self.camera_width, self.camera_height

                # 等比例填满 + 居中裁剪
                scale = max(target_w / frame_w, target_h / frame_h)
                new_w = int(frame_w * scale)
                new_h = int(frame_h * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                left = (new_w - target_w) // 2
                top = (new_h - target_h) // 2
                right = left + target_w
                bottom = top + target_h
                img = img.crop((left, top, right, bottom))
                
                imgtk = ImageTk.PhotoImage(image=img)
                self.lbl_camera.imgtk = imgtk
                self.lbl_camera.configure(image=imgtk)
                
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.root.after(30, self.update_camera_feed)

    def log_message(self, message, is_clear=False):
        self.root.after(0, self._update_text, message, is_clear)

    def _update_text(self, message, is_clear):
        self.text_result.config(state=tk.NORMAL)
        if is_clear:
            self.text_result.delete(1.0, tk.END)
        self.text_result.insert(tk.END, message + "\n")
        self.text_result.see(tk.END)
        self.text_result.config(state=tk.DISABLED)

    def set_button_state(self, state):
        def update():
            if state == tk.DISABLED:
                self.btn_analyze.config(state=tk.DISABLED, bg="#D1C7BD", text="分析中...")
            else:
                self.btn_analyze.config(state=tk.NORMAL, bg=self.color_primary, text="开始识别分析")
        self.root.after(0, update)

    def start_analysis_thread(self):
        if self.current_frame is None:
            messagebox.showwarning("提示", "摄像头未就绪，无法获取画面")
            return
        if self.is_analyzing: return
            
        self.is_analyzing = True
        self.set_button_state(tk.DISABLED)
        self.log_message("正在启动识别分析流程...", is_clear=True)
        
        frame_to_analyze = self.current_frame.copy()
        
        thread = threading.Thread(target=self.run_analysis, args=(frame_to_analyze,))
        thread.daemon = True
        thread.start()

    def run_analysis(self, frame):
        try:
            user_status = self.health_var.get()
            if self.model is None:
                self.log_message("首次加载 YOLO 模型，请稍候...")
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            
            self.log_message("YOLO 推理中...")
            results = self.model.predict(source=frame, device='cpu', save=False, conf=0.25)
            
            obj_list = []
            for res in results:
                for cls_id in res.boxes.cls:
                    obj_list.append(self.model.names[int(cls_id)])

            if not obj_list:
                self.log_message("未识别到任何食物，请调整画面后重试")
                return

            target_food = obj_list[0]
            now_t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self.log_message("-" * 60)
            self.log_message(f"识别完成\n食物: {target_food}\n时间: {now_t}\n状态: {user_status}")
            self.log_message("-" * 60)
            
            self.log_message("\n正在调用 DeepSeek 生成营养分析...")
            ans = ask_deepseek(target_food, now_t, user_status)

            self.log_message("\n================ 分析结果 ================\n")
            self.log_message(ans)
            self.log_message("\n================== 结束 ==================")

        except Exception as e:
            self.log_message(f"\n运行出错:\n{str(e)}")
        finally:
            self.is_analyzing = False
            self.set_button_state(tk.NORMAL)
            
    def on_closing(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LongxinDietAnalyzerApp(root)
    try:
        root.state('zoomed')
    except:
        try:
            root.attributes('-zoomed', True)
        except:
            pass
    root.mainloop()
