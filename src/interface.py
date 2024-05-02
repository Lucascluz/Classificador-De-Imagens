import os
import cv2
import torch

import numpy as np
import tkinter as tk
import torch.nn as nn
import torch.nn.functional as F

from tkinter import filedialog
from tkinter import ttk
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from PIL import Image, ImageTk
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split

global zoom_factor

class TireNet(nn.Module):
        
    def __init__(self, printsize):
            
        super().__init__()
            
        # For printing size for each layer
        self.printsize = printsize
            
        # Model architecture (CNN Layers)
        
        self.con1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.con2 = nn.Conv2d(32, 64, 5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.con3 = nn.Conv2d(64, 128, 5, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        # Maxpool 
        self.pool = nn.MaxPool2d(2)
        
        # FFN layer
        self.fnn1 = nn.Linear(4608, 50)
        self.bnfnn = nn.BatchNorm1d(50)
        self.dropfnn = nn.Dropout(0.5)
        self.output = nn.Linear(50, 1)
        
    def forward(self, x):
            
        # CNN Layer 1
        if self.printsize: print(f"Input shape is {x.shape} before go to con1")
        x = F.relu(self.con1(x))
        x = self.bn1(x)
        x = self.pool(x)
        
        # CNN Layer 2 with maxpool
        if self.printsize: print(f"Input shape is {x.shape} after con1")
        x = F.relu(self.con2(x))
        x = self.bn2(x)
        x = self.pool(x)
            
        # CNN Layer 3
        if self.printsize: print(f"Input shape is {x.shape} after con2 and maxpool")
        x = F.relu(self.con3(x))
        x = self.bn3(x)
        x = self.pool(x)
            
        # Reshape x into vector
        x = x.view(x.shape[0], -1)
        if self.printsize: print(f"x size after reshape is {x.shape}")
                
        # FNN Layer 1 - 2
        x = F.relu(self.fnn1(x))
        x = self.bnfnn(x)

        return self.output(x)

class ImageViewerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")

        self.frame = tk.Frame(self.master)
        self.frame.pack(fill=tk.BOTH, expand=True,)

        self.canvas = tk.Canvas(self.frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,)
        
        self.image_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_frame, anchor=tk.NW, )

        self.image_label = tk.Label(self.image_frame, )
        self.image_label.pack()
            
        self.scrollbarVertical = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbarVertical.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbarVertical.set)

        self.scrollbarHorizontal = ttk.Scrollbar(self.master, orient=tk.HORIZONTAL, command=self.canvas.xview, )
        self.scrollbarHorizontal.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.configure(xscrollcommand=self.scrollbarHorizontal.set)

        # Criando um frame para os botões adicionais
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack()
        
        self.open_button = tk.Button(self.button_frame, text="Open Image", command=self.open_image, width=20, height=2)
        self.open_button.pack(side=tk.LEFT)
        
        self.zoom_plus_button = tk.Button(self.button_frame, text="+", width=6, height=2, state="disabled")
        self.zoom_plus_button.pack(side=tk.LEFT)
        
        self.zoom_minus_button = tk.Button(self.button_frame, text="-", width=6, height=2, state="disabled")
        self.zoom_minus_button.pack(side=tk.LEFT)
        
        self.classify_button = tk.Button(self.button_frame, text="Classification", width=20, height=2, state = "disabled")
        self.classify_button.pack(side=tk.LEFT)
        
    def place_class(self, file_path):
        global zoom_factor
        
        image = cv2.imread(file_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Atualizando o tamanho da imagem
        img_width = int(image_pil.width * zoom_factor)
        img_height = int(image_pil.height * zoom_factor)

        # Redimensionando a imagem
        image_resized = image_pil.resize((img_width, img_height))
        photo = ImageTk.PhotoImage(image_resized)
        
        # Limpa o canvas antes de adicionar a nova imagem
        self.canvas.delete("all")

        # Calcula as coordenadas para centralizar a imagem
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = photo.width()
        image_height = photo.height()
        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        # Adiciona a imagem ao canvas centralizada
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image = photo)
        self.canvas.image = photo
        
        self.canvas.config(scrollregion=self.canvas.bbox("all")) 
        
    def place_image(self, image_pil):
        global zoom_factor
        
        # Atualizando o tamanho da imagem
        img_width = int(image_pil.width * zoom_factor)
        img_height = int(image_pil.height * zoom_factor)

        # Redimensionando a imagem
        image_resized = image_pil.resize((img_width, img_height))
        photo = ImageTk.PhotoImage(image_resized)
        
        # Limpa o canvas antes de adicionar a nova imagem
        self.canvas.delete("all")

        # Calcula as coordenadas para centralizar a imagem
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = photo.width()
        image_height = photo.height()
        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        # Adiciona a imagem ao canvas centralizada
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image = photo)
        self.canvas.image = photo
        
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        self.zoom_plus_button.config(state="active")
        self.zoom_minus_button.config(state="active")    
        self.classify_button.config(state="active", text="Classification", command= lambda: self.classify_image(image_pil))
            
    def open_image(self):
        file_path = filedialog.askopenfilename()
        
        if file_path:
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil_base = Image.fromarray(image_rgb)

            # Define o fator de zoom para o valor padrão
            global zoom_factor
            zoom_factor = 1.0

            self.place_image(image_pil_base)
            
            self.open_button.config(text="Change Image")
            self.zoom_plus_button.config(state="active", command = lambda: self.zoom_plus(image_pil_base))
            self.zoom_minus_button.config(state="active", command = lambda: self.zoom_minus(image_pil_base))
            self.classify_button.config(state="active", text="Classification",command= lambda: self.classify_image(image_pil_base))
            
    def zoom_plus(self, image_pil_base):
        # Aumenta o fator de zoom em 10%
        global zoom_factor
        zoom_factor = zoom_factor + 0.1

        self.place_image(image_pil_base)

        
    def zoom_minus(self, image_pil_base):
        # Aumenta o fator de zoom em 10%
        global zoom_factor
        zoom_factor = zoom_factor - 0.1

        self.place_image(image_pil_base)
        
    def preprocessing(self, image_pil_base):
        # Redimensione a imagem para ser compatível com a forma desejada
        image_resized = image_pil_base.resize((224, 224))
        
        # Normalize a imagem
        image_processed = np.array(image_resized) / np.max(image_resized)
        
        # Remodele a imagem para a forma desejada
        image_processed = torch.tensor(image_processed.reshape(1, 3, 224, 224)).float()
        
        return image_processed
        
    def classify_image(self, image_pil):
        
        input_image = self.preprocessing(image_pil) 
        # Carregue o modelo salvo
        model = TireNet(printsize = False)  # Crie uma instância da sua rede neural
        model.load_state_dict(torch.load('modelo_predicao_pth'))
        model.eval()  # Coloque o modelo no modo de avaliação
        
        # Suponha que 'input_image' seja a imagem de entrada
        with torch.no_grad():
            output = model(input_image)
            predicted_class = torch.argmax(output).item()
        if(predicted_class == 0):
            self.place_class("assets/cracked.jpeg")
        else:
            self.place_class("assets/normal.jpeg")
            
        self.zoom_plus_button.config(state="disabled")
        self.zoom_minus_button.config(state="disabled")    
        self.classify_button.config(state="active", text="Image", command= lambda: self.place_image(image_pil))
            
def main():
    # Cria a janela principal da aplicação
    root = tk.Tk()
    
    # Define o estado inicial da janela como maximizado
    root.state('zoomed')  # 'zoomed' maximiza a janela na maioria dos sistemas
    
    # Inicializa a aplicação e inicia o loop principal da interface gráfica
    app = ImageViewerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
