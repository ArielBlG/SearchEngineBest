import os
import time

import pandas as pd
from reader import ReadFile
from configuration import ConfigClass
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher
import utils
from gensim.models import KeyedVectors


# DO NOT CHANGE THE CLASS NAME
class SearchEngine:

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation, but you must have a parser and an indexer.
    def __init__(self, config=None):
        self._config = config
        self._indexer = Indexer(config)
        self.load_model(os.path.join('.', 'model'))
        self._model = self._config.model_dir
        # self.load_precomputed_model(os.path.join('.', 'model'))
        self._parser = Parse(model=self._model)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def build_index_from_parquet(self, fn):
        """
        Reads parquet file and passes it to the parser, then indexer.
        Input:
            fn - path to parquet file
        Output:
            No output, just modifies the internal _indexer object.
        """
        start = time.time()
        df = pd.read_parquet(fn, engine="pyarrow")
        documents_list = df.values.tolist()
        # Iterate over every document in the file
        number_of_documents = 0
        for idx, document in enumerate(documents_list):
            # parse the document
            for parsed_document in self._parser.parse_doc(document):
                number_of_documents += 1
                self._indexer.add_new_doc(parsed_document)
        # self._indexer.compute_weights_per_doc()
        # self._indexer.save_index("idx_bench")
        print('Finished parsing and indexing.')
        print(f'finished parsing and indexing best in {time.time()-start} ms')

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        self._indexer.load_index(fn)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_precomputed_model(self, model_dir=None):
        """
        Loads a pre-computed model (or models) so we can answer queries.
        This is where you would load models like word2vec, LSI, LDA, etc. and 
        assign to self._model, which is passed on to the searcher at query time.
        """
        # self._model = KeyedVectors.load_word2vec_format(model_dir + '/word2vec_model.bin', binary=True)
        pass

    def load_model(self, model_dir=None):
        """

        """
        model = KeyedVectors.load_word2vec_format(model_dir + '/word2vec_model.bin', binary=True)
        self._config.model_dir = model

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query):
        """
        Executes a query over an existing index and returns the number of
        relevant docs and an ordered list of search results.
        Input:
            query - string.
        Output:
            A tuple containing the number of relevant search results, and
            a list of tweet_ids where the first element is the most relavant
            and the last is the least relevant result.
        """
        searcher = Searcher(self._parser,
                            self._indexer,
                            model=self._model,
                            word_net=True,
                            w2v=True,
                            spell_checker=True)
        return searcher.search_2(query)


def main():
    start = time.time()
    search_engine = SearchEngine()
    search_engine.build_index_from_parquet('data/benchmark_data_train.snappy.parquet')
    end = time.time()
    print(end - start)
    first_query = r"Dr. Anthony Fauci wrote in a 2005 paper published in Virology Journal that hydroxychloroquine was effective in treating SARS."
    second_query = r"The seasonal flu kills more people every year in the U.S. than COVID-19 has to date."
    four_query = r"The coronavirus pandemic is a cover for a plan to implant trackable microchips and that the Microsoft co-founder Bill Gates is behind it"
    seven_query = r"Herd immunity has been reached."
    eight_query = r"Children are “almost immune from this disease.”"
    print(search_engine.search(first_query))
    print(search_engine.search(second_query))
    print(search_engine.search(four_query))
    print(search_engine.search(seven_query))
    print(search_engine.search(eight_query))


if __name__ == '__main__':
    main()
