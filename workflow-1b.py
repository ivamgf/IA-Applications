import nltk
import pandas as pd
import os
import webbrowser
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
        if idx not in frequency_distribution:
            frequency_distribution[idx] = FreqDist()
        frequency_distribution[idx].update(word_tokens)

# Preparar palavras para a WordCloud
unique_words = set()
for idx, freq_dist in frequency_distribution.items():
    for word, freq in freq_dist.items():
        if freq > 1:
            unique_words.add(word)

# Criar a WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(unique_words))

# Visualizar a WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# Salvar a WordCloud como uma imagem
wordcloud_image_path = 'output/wordcloud.png'
plt.savefig(wordcloud_image_path, bbox_inches='tight')

# Abrir a WordCloud salva
webbrowser.open('file://' + os.path.abspath(wordcloud_image_path))

# Message
print(f"WordCloud salva com sucesso em {wordcloud_image_path}")
