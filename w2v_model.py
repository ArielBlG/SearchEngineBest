import pandas as pd
from gensim.models import Word2Vec
import multiprocessing
from gensim.models.phrases import Phrases, Phraser
path = ""
df = pd.read_csv(path)
df = df.drop(columns = ['Unnamed: 0'])
sent = [str(row).split() for row in df['body']]
phrases = Phrases(sent, min_count=30, progress_per=10000)
bigram = Phraser(phrases)
sentences = bigram[sent]
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores-1)
from time import time
t = time()

w2v_model.build_vocab(sentences, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
t = time()

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
w2v_model.init_sims(replace=True)
word_vectors = w2v_model.wv
word_vectors.save_word2vec_format("word2vec_model.bin", binary=True)
w2v_model.save("word2vec_model.bin", binary=True)
