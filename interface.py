import customtkinter as ctk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import torch
from torch import nn
from torchvision.transforms import ToTensor, Normalize, Compose, Resize, CenterCrop
from efficientnet_pytorch import EfficientNet
from torchvision import models
import os
import time

global zoom_factor

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        efficient_net_output_dim = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Identity()  # Remove the final classification layer
        self.fc_combined = nn.Linear(efficient_net_output_dim, num_classes)

    def forward(self, images):
        efficient_net_out = self.efficient_net(images)
        efficient_net_out = torch.flatten(efficient_net_out, 1)
        out = self.fc_combined(efficient_net_out)
        return out

class CustomMessageBox(ctk.CTkToplevel):
    def __init__(self, master, title, message):
        super().__init__(master)
        
        self.title(title)
        self.geometry("300x150")
        self.resizable(False, False)

        self.label = ctk.CTkLabel(self, text=message, wraplength=250, justify="center")
        self.label.pack(pady=20)

        self.button = ctk.CTkButton(self, text="OK", command=self.destroy)
        self.button.pack(pady=10)
        
        # Make the message box modal
        self.grab_set()
        self.transient(master)

class ImageViewerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")
        self.master.geometry("800x600")
        self.master.resizable(True, True)

        self.frame = ctk.CTkFrame(self.master)
        self.frame.pack(fill=ctk.BOTH, expand=True)

        self.canvas = ctk.CTkCanvas(self.frame)
        self.canvas.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)
        
        self.image_frame = ctk.CTkFrame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_frame, anchor=ctk.NW)

        self.image_label = ctk.CTkLabel(self.image_frame)
        self.image_label.pack()
            
        self.scrollbarVertical = ctk.CTkScrollbar(self.frame, orientation=ctk.VERTICAL, command=self.canvas.yview)
        self.scrollbarVertical.pack(side=ctk.RIGHT, fill=ctk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbarVertical.set)

        self.scrollbarHorizontal = ctk.CTkScrollbar(self.master, orientation=ctk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbarHorizontal.pack(side=ctk.BOTTOM, fill=ctk.X)
        self.canvas.configure(xscrollcommand=self.scrollbarHorizontal.set)

        self.button_frame = ctk.CTkFrame(self.master)
        self.button_frame.pack()

        self.button_info = self.create_buttons()
        
        self.image_pil_base = None
        
    def create_buttons(self):
        button_info = [
            ("Open Image", self.open_image),
            ("+", self.zoom_plus),
            ("-", self.zoom_minus),
            ("Classify", self.eficinetBinaryClassification),
        ]

        for text, command in button_info:
            state = "active"
            btn = ctk.CTkButton(self.button_frame, text=text, command=command, width=100, height=50, state=state)
            btn.pack(side=ctk.LEFT)
            setattr(self, f"{text.replace(' ', '_').lower()}_button", btn)
            
        return button_info

    def place_graph(self, photo):
        self.canvas.delete("all")

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = photo.width()
        image_height = photo.height()
        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=photo)
        self.canvas.image = photo
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))    

    def place_image(self, image_pil):
        img_width = int(image_pil.width * zoom_factor)
        img_height = int(image_pil.height * zoom_factor)
        image_resized = image_pil.resize((img_width, img_height))
        photo = ImageTk.PhotoImage(image_resized)

        self.place_graph(photo)
            
    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_pil_base = Image.fromarray(image_rgb)

            global zoom_factor
            zoom_factor = 1.0

            self.place_image(self.image_pil_base)
            
            self.open_image_button.configure(text="Change Image")

    def zoom_plus(self):
        global zoom_factor
        zoom_factor += 0.1
        self.place_image(self.image_pil_base)

    def zoom_minus(self):
        global zoom_factor
        zoom_factor -= 0.1
        self.place_image(self.image_pil_base)

    def update_image(self, cvt_color_code, mode):
        if self.image_pil_base:
            image_pil_rgb = self.image_pil_base.convert('RGB')
            matrix_rgb = np.array(image_pil_rgb)
            matrix_converted = cvt_color_code == 'convert' and np.array(image_pil_rgb.convert(mode)) or cv2.cvtColor(matrix_rgb, cvt_color_code)
            image_pil_converted = Image.fromarray(matrix_converted)
            self.place_image(image_pil_converted)

    def eficinetBinaryClassification(self):
        start_time = time.time()
        model_path = 'efficientnet_binary_fine_tuned.pth'

        preprocess = Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = self.image_pil_base.convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)

        model = models.efficientnet_b0()
        num_features = model.classifier[1].in_features
        model.fc = nn.Linear(num_features, 2)

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))        
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)
        class_labels = ["Normal", "Cracked"]
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_labels[predicted_idx.item()]
        prediction_time = time.time() - start_time
        
        # Use the CustomMessageBox for consistent theme
        CustomMessageBox(self.master, "Classification", f"Prediction: {predicted_label}\nTime: {prediction_time:.2f}s")
            
    
def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = ImageViewerApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
