import nltk
import pandas as pd
import os
import webbrowser
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

# Downloads
nltk.download('punkt')

# Path Input
path = 'input/Table-Sentences.csv'

# Read CSV
try:
    df = pd.read_csv(path)
except FileNotFoundError:
    print(f"Arquivo CSV não encontrado: {path}")
    exit()

content = df['Pre-processed']

# Tokenização por sentenças
sentences_list = []
for idx, sentence in enumerate(content):
    sentence_tokens = sent_tokenize(sentence)
    sentences_list.append((idx, sentence_tokens))

# Tokenização por palavras
tokenized_sentences = []
for idx, sentence_tokens in sentences_list:
    word_tokens = [word_tokenize(sent) for sent in sentence_tokens]
    tokenized_sentences.append((idx, word_tokens))

# Frequência de distribuição
frequency_distribution = {}
for idx, sentence_tokens in tokenized_sentences:
    for word_tokens in sentence_tokens:
        frequency_distribution[idx] = FreqDist(word_tokens)

# Caminhos para os arquivos de saída
path_html = 'output/output.html'
path_txt = 'output/output.txt'

# Save HTML
with open(path_html, 'w', encoding='utf-8') as file_html:
    for idx, sentence_tokens in tokenized_sentences:
        file_html.write(f"<p>Sentence {idx}:</p>")
        for word_tokens in sentence_tokens:
            file_html.write("<ul>")
            for word in word_tokens:
                file_html.write(f"<li>{word} - Frequência: {frequency_distribution[idx][word]}</li>")
            file_html.write("</ul>")

webbrowser.open('file://' + os.path.abspath(path_html))

# Save TXT
with open(path_txt, 'w', encoding='utf-8') as file_txt:
    for idx, sentence_tokens in tokenized_sentences:
        file_txt.write(f"Sentence {idx}:\n")
        for word_tokens in sentence_tokens:
            for word in word_tokens:
                file_txt.write(f"{word} - Frequência: {frequency_distribution[idx][word]}\n")

# Message
print(f"Arquivos HTML e TXT salvos com sucesso em {path_html} e {path_txt}")
