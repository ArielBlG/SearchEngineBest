import os
import time

import pandas as pd
from reader import ReadFile
from configuration import ConfigClass
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher
from GUI import GUI
import utils
from gensim.models import KeyedVectors


# DO NOT CHANGE THE CLASS NAME
class SearchEngine:

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation, but you must have a parser and an indexer.
    def __init__(self, config=None):
        self._config = config
        self._indexer = Indexer(config)
        self._model = None
        self.load_model(self._config.model_dir)
        self.load_precomputed_model(self._config.model_dir)
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
        self._indexer.save_index('idx_bench')
        print('Finished parsing and indexing.')
        print(f'Finished parsing and indexing best in {time.time()-start} ms')

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
        self._model = KeyedVectors.load_word2vec_format(self._config.model_dir, binary=True)
        # pass

    def load_model(self, model_dir=None):
        """

        """
        # model = KeyedVectors.load_word2vec_format(os.path.join(model_dir, "word2vec_model.bin"), binary=True)
        # if model.vocab:
        #     print(f"model loaded succefully with {len(model.vocab.keys())} words")

        self._config.model_dir = os.path.join(os.path.join('.', 'model'), "word2vec_model.bin")

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
    conf = ConfigClass()
    model_url = conf.get_model_url()
    if model_url is not None and conf.get_download_model():
        dest_path = 'model.zip'
        utils.download_file_from_google_drive(model_url,dest_path)
        if not os.path.exists(os.path.join('.', 'model/205417637')):
            os.mkdir(os.path.join('.', 'model/205417637'))
        utils.unzip_file(dest_path, os.path.join('.', 'model/205417637'))
    path = conf.get__corpusPath()
    pd.options.display.max_colwidth = 10000
    df = pd.read_parquet(path, engine="pyarrow")

    start = time.time()
    gui = GUI()
    gui.window.Read(timeout=0)
    search_engine = SearchEngine(config=conf)
    search_engine.build_index_from_parquet(conf.get__corpusPath())
    end = time.time()
    print(end - start)
    gui.switchWindow()
    while(True):
        try:
            type, res = gui.runGUI()
            if type == 1 and res == "Exit":
                break
            if type == 1 and res != None and res != '':
                answer = search_engine.search(res)
                gui.updateAnswers(answer[1])
            if type == 2 and res != None and res != []:
                tweet = list(df.loc[df['tweet_id'] == res[0]]['full_text'])[0]
                gui.setDocText(tweet)
        except Exception as e:
            print(e)
    gui.window.close()

if __name__ == '__main__':
    main()
