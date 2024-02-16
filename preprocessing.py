import pandas as pd
import shutil
import os

# Carregar o CSV para um DataFrame do pandas
df = pd.read_csv('data/valid/_classes.csv', header=None)  # Definir header=None para indicar que não há cabeçalho no arquivo CSV

# Criar diretórios para as três categorias
categorias = ['fresh', 'half-fresh', 'spoiled']
for categoria in categorias:
    os.makedirs(categoria, exist_ok=True)

# Iterar sobre as linhas do DataFrame e mover os arquivos para os diretórios apropriados
for index, row in df.iterrows():
    imagem = row[0]  # Índice 0 corresponde ao nome do arquivo
    fresh = row[1].strip()   # Remover espaços em branco extras e converter para string
    half_fresh = row[2].strip()  # Remover espaços em branco extras e converter para string
    spoiled = row[3].strip()  # Remover espaços em branco extras e converter para string
    
    # Verificar se a imagem possui uma classe válida
    if fresh == '1':
        destino = 'fresh/'
    elif half_fresh == '1':
        destino = 'half-fresh/'
    elif spoiled == '1':
        destino = 'spoiled/'
    else:
        print(f"A imagem {imagem} não possui uma classe válida.")
        continue
    
    origem = 'data/valid/' + imagem  # Caminho para a imagem original
    destino_completo = destino + imagem  # Caminho para onde a imagem deve ser movida
    
    shutil.move(origem, destino_completo)  # Movendo a imagem

print("Imagens classificadas com sucesso.")
