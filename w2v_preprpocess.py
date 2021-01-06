import os

import pandas as pd
from tqdm import tqdm
from parse_patterns import *

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
        tweet_id = document[0]
        full_text = document[2]
        doc = pd.DataFrame(list(zip([tweet_id], [full_text])), columns=["Tweet_id", "body"])
        doc['body'] = doc['body'].apply(lambda x: expand_contractions(x))
        doc['body'] = doc['body'].apply(lambda x: clean_text(x))
        doc['body'] = doc['body'].apply(lambda x: re.sub(' +', ' ', x))
        doc['body'] = doc['body'].apply(
            lambda x: ' '.join([token.lemma_ for token in list(nlp(x)) if (not token.is_stop)]))
        return doc


def create_corpus():
    df = pd.DataFrame(columns=["Tweet_id", "body"])
    return df


def read_data(corpus_path=''):
    index_data = 7
    corpus = create_corpus()
    parser = PreProcessW2V()
    config = ConfigClass()
    config.corpusPath = corpus_path
    pp_w2v = PreProcessW2V()
    reader = ReadFile(corpus_path=config.get__corpusPath())
    number_of_documents = 0
    counter = 0
    # for filename in os.listdir(config.get__corpusPath()):
    #     if filename == ".DS_Store":
    #         continue
    #     for par_name in os.listdir(config.get__corpusPath() + '/' + filename):
    #         corpus = create_corpus()
    #         if par_name == ".DS_Store":
    #             continue
    #         doc_list = reader.read_file(file_name=filename + '/' + par_name)
    #         if counter < 7:
    #             counter += 1
    #             continue
    #         print(counter)
    #         for document in tqdm(doc_list, desc="Parsing + Indexing"):
    doc_list = reader.read_file("benchmark_data_train.snappy.parquet")
    for document in tqdm(doc_list, desc="Parsing + Indexing"):
        corpus = pd.concat([corpus, parser.parse_document(document)])
    path = "/Users/ariel-pc/Documents/שנה ג/IR/SearchEngine/SearchEngine/w2v_data/"
    path += str(21) + ".csv"
    index_data += 1
    corpus.to_csv(path)
            # for parsed_document in parser.parse_doc(document):
            #     # index the document data
            #     number_of_documents += 1
            #     indexer.add_new_doc(parsed_document)


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


def main():
    # read_data()
    merge_csvs()


if __name__ == "__main__":
    main()
