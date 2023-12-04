import pandas as pd
import os
import webbrowser

# Path Input
path = 'input/Table-Sentences.csv'

# Read CSV
try:
    df = pd.read_csv(path)
except FileNotFoundError:
    print(f"Arquivo CSV não encontrado: {path}")
    exit()

content = df['Pre-processed']

sentences = content.str.split('.').explode().reset_index(name='Sentenca')

# Caminhos para os arquivos de saída
path_html = 'output/output.html'
path_txt = 'output/output.txt'

# Save HTML
with open(path_html, 'w', encoding='utf-8') as file_html:
    file_html.write(sentences.to_html(index=False))

webbrowser.open('file://' + os.path.abspath(path_html))

# Save TXT
with open(path_txt, 'w', encoding='utf-8') as file_txt:
    file_txt.write(str(sentences.to_string(index=False)))

# Message
print(f"Arquivos HTML e TXT salvos com sucesso em {path_html} e {path_txt}")


