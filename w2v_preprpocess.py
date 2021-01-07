import os

import pandas as pd
from tqdm import tqdm
from parse_patterns import *
from utils import *

from configuration import ConfigClass
from reader import ReadFile
import spacy

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
nlp.max_length = 5000000

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


def clean_text(text):
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub('[^a-z]', ' ', text)
    return text


# Function for expanding contractions
def expand_contractions(text, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, text)


class PreProcessW2V:

    def __init__(self):
        pass

    def parse_document(self, document):
        path = '/Users/ariel-pc/Desktop/min_inverted.pickle'
        inv_index = load_obj(path)
        tweet_id = document[0]
        full_text = document[2]
        doc = pd.DataFrame(list(zip([tweet_id], [full_text])), columns=["Tweet_id", "body"])
        doc['body'] = doc['body'].apply(lambda x: expand_contractions(x))
        doc['body'] = doc['body'].apply(lambda x: clean_text(x))
        doc['body'] = doc['body'].apply(lambda x: re.sub(' +', ' ', x))
        doc['body'] = doc['body'].apply(
            lambda x: ' '.join([token.lemma_ for token in list(nlp(x, disable=["tagger", "parser", "ner"])) if (not token.is_stop)]))
        doc['body'] = doc['body'].apply(lambda x: ' '.join([token for token in x.split() if token not in inv_index]))
        return doc


def create_corpus():
    df = pd.DataFrame(columns=["Tweet_id", "body"])
    return df


def read_data(corpus_path=''):
    index_data = 0
    corpus = create_corpus()
    parser = PreProcessW2V()
    config = ConfigClass()
    config.corpusPath = corpus_path
    pp_w2v = PreProcessW2V()
    reader = ReadFile(corpus_path=config.get__corpusPath())
    number_of_documents = 0
    counter = 0
    path = "/Users/ariel-pc/Documents/שנה ג/IR/SearchEngine/SearchEngine/w2v_data_new/"
    for filename in os.listdir(config.get__corpusPath()):
        if filename == ".DS_Store":
            continue
        for par_name in os.listdir(config.get__corpusPath() + '/' + filename):
            corpus = create_corpus()
            if par_name == ".DS_Store":
                continue
            doc_list = reader.read_file(file_name=filename + '/' + par_name)

            print(counter)
            for document in tqdm(doc_list, desc="Parsing + Indexing"):
                corpus = pd.concat([corpus, parser.parse_document(document)])
        path += str(index_data) + ".csv"
        index_data += 1
        corpus.to_csv(path)


def merge_csvs():
    fout = open("/Users/ariel-pc/Documents/שנה ג/IR/SearchEngine/SearchEngine/w2v_data/second_model.csv", "a")
    for line in open("/Users/ariel-pc/Documents/שנה ג/IR/SearchEngine/SearchEngine/w2v_data/11.csv"):
        fout.write(line)
    for num in range(12, 22):
        f = open("/Users/ariel-pc/Documents/שנה ג/IR/SearchEngine/SearchEngine/w2v_data/" + str(num) + ".csv")
        # f.next()  # skip the header
        lines = f.readlines()[1:]
        for line in lines:
            fout.write(line)
        f.close()  # not really needed
    fout.close()


def test_inverted_idx():
    path = '/Users/ariel-pc/Desktop/corpus_inv_idx.pickle'
    object_to_open = load_obj(path)
    sorted_dict = dict(sorted(object_to_open.items(), key=lambda item: item[1]))
    new_dict = [(key,value) for key,value in sorted_dict.items() if value > 15]
    new_dict = dict(new_dict)
    path = '/Users/ariel-pc/Desktop/min_inverted.pickle'
    with open(path , 'wb') as f:
        pickle.dump(path, f, pickle.HIGHEST_PROTOCOL)


def main():
    read_data()
    # merge_csvs()
    # test_inverted_idx()


if __name__ == "__main__":
    main()
