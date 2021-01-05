from sortedcontainers import SortedDict
import numpy as np

from ranker import Ranker
import utils
from gensim.models import KeyedVectors

# DO NOT MODIFY CLASS NAME
class Searcher:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit. The model 
    # parameter allows you to pass in a precomputed model that is already in 
    # memory for the searcher to use such as LSI, LDA, Word2vec models. 
    # MAKE SURE YOU DON'T LOAD A MODEL INTO MEMORY HERE AS THIS IS RUN AT QUERY TIME.
    def __init__(self, parser, indexer, model=None):
        self._parser = parser
        self._indexer = indexer
        self._ranker = Ranker()
        self._model = model
        self.wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query, k=None):
        """ 
        Executes a query over an existing index and returns the number of 
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
        Output:
            A tuple containing the number of relevant search results, and 
            a list of tweet_ids where the first element is the most relavant 
            and the last is the least relevant result.
        """
        query_as_list = self._parser.parse_sentence(query)
        # query_as_list = self.expand_query(query_as_list, sim_to_expand=0.75)
        query_as_list = self._parser.get_lemma_text(query_as_list)
        vector_query = self.get_vector_query(query_as_list)
        w2v_vector = self.get_w2v_vector_query(query_as_list)  # TODO: remove for w2v
        relevant_docs, doc_id_set = self._relevant_docs_from_posting(query_as_list)
        n_relevant = len(relevant_docs)
        tweets_dict = self._indexer.get_tweets_dict()
        ranked_doc_ = Ranker.rank_relevant_docs(relevant_docs,
                                                doc_set=doc_id_set,
                                                vector=np.array(list(vector_query)),
                                                tweets_dict=tweets_dict,
                                                w2v_vector=w2v_vector)
        ranked_doc_ids = [doc_id[0] for doc_id in ranked_doc_]
        return n_relevant, ranked_doc_ids

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def _relevant_docs_from_posting(self, query_as_list):
        """
        This function loads the posting list and count the amount of relevant documents per term.
        :param query_as_list: parsed query tokens
        :return: dictionary of relevant documents mapping doc_id to document frequency.
        """
        relevant_docs = {}
        doc_id_set = set()
        for term in query_as_list:
            posting_list = self._indexer.get_term_posting_list(term)
            doc_id_list = list(map(lambda item: item[0], posting_list))
            doc_id_set.update(doc_id_list)
            relevant_docs[term] = posting_list
            # for doc_id, tf, appearance_num in posting_list:
            #     df = relevant_docs.get(doc_id, 0)
            #     relevant_docs[term] = df + 1
        return relevant_docs, doc_id_set

    def get_vector_query(self, query):
        """
        :param query: query from user
        :return: list representing the query as a vector after computed tf_idf
        """
        dic_term = SortedDict()
        max_occurrence = 1
        for term in query:
            if term in dic_term:
                dic_term[term] += 1
                if dic_term[term] > max_occurrence:
                    max_occurrence = dic_term[term]
            else:
                dic_term[term] = 1
        num_of_docs = self._indexer.get_number_of_docs()
        for key, value in dic_term.items():
            # dic_term[key] = value / max_occurrence \
            #                 * np.log10(num_of_docs / self._indexer.get_term_from_inverted(key))
            dic_term[key] = value / max_occurrence
        return dic_term.values()

    def get_w2v_vector_query(self, query_list):
        num_of_vectors = 0
        query_vector = np.zeros(300)
        for term in query_list:
            try:
                vector = self.wv[term]
                num_of_vectors += 1
                query_vector += vector
            except:
                pass
        return query_vector / num_of_vectors

    def expand_query(self, query_as_list, sim_to_expand=0.7):
        new_query_list = []
        for term in query_as_list:
            try:
                list_out = self.wv.most_similar(term)
                new_query_list += [tup[0] for tup in list_out if tup[1] > sim_to_expand]
            except:
                pass
        return query_as_list + new_query_list
