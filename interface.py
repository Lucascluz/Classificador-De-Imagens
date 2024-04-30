import os
import cv2
import io
import mahotas
import tempfile

import numpy as np
import tkinter as tk

from tkinter import filedialog
from tkinter import ttk
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from PIL import Image, ImageTk
from skimage import color
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
import tensorflow as tf

global zoom_factor

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
        
        self.classify_button = tk.Button(self.button_frame, text="Classify", width=20, height=2, state = "disabled")
        self.classify_button.pack(side=tk.LEFT)
        
    def place_graph(self, photo):
        global zoom_factor
        
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
            self.classify_button.config(state="active", command= lambda: self.classify_image(image_pil_base))
            
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
        
    def classify_image(self, image_pil):
        # Redimensione a imagem, se necessário
        im_resize = image_pil.resize((416, 416))

        # Converta a imagem redimensionada para uma matriz
        image_array = tf.keras.preprocessing.image.img_to_array(im_resize)

        # Expanda as dimensões para corresponder ao formato de entrada do modelo
        image_input = tf.expand_dims(image_array, 0)

        # Carregue o modelo pré-treinado
        model = tf.keras.models.load_model('model.keras')

        # Faça uma previsão
        prediction = model.predict(image_input)[0][0]

        # Determine a classificação
        resultado = 'rachada' if prediction < 0.5 else 'normal'

        print("Previsão: {prediction:.4f} | {resultado}")
            
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
