import time

from sortedcontainers import SortedDict
import numpy as np
from nltk.corpus import wordnet as wn
from ranker import Ranker
from spellchecker import SpellChecker
spell = SpellChecker()
import utils
from gensim.models import KeyedVectors


def spell_checker_search(query_list):
    """
    This function tries to fix misspelling terms inside a query
    :param query_list: list of terms representing the query's terms
    :return: new query list after fixing misspells
    """
    new_query_list = []
    for term in query_list:
        misspelled = spell.unknown([term])
        if not misspelled:
            new_query_list.append(term)
        else:
            new_query_list.append(spell.correction(misspelled.pop()))
            new_query_list.append(term)
    return new_query_list

def _get_words_for_expansion(ranked_doc_, query_as_list, tweets_dic, k=1):
    """
    This function returns a new word of each word inside query_as_list
    :param ranked_doc_: dataframe containing docs and their ranks from first check
    :param query_as_list: list of terms representing the query's terms
    :param tweets_dic: tweets dictionary
    :param k: number of words to expand, defaults to 1
    :return: new query list after fixing misspells
    """
    bag_of_words = list(
        map(lambda docID: list(tweets_dic[docID][4].keys()), list(map(lambda doc: doc[0], ranked_doc_))))
    bag_of_words = sorted(list(set([item for sublist in bag_of_words for item in sublist])))
    matrix = {}
    for word in bag_of_words:
        matrix[word] = dict(map(lambda word: (word, 0), bag_of_words))

    for doc_id in ranked_doc_:
        for term_i in tweets_dic[doc_id[0]][4]:
            for term_j in tweets_dic[doc_id[0]][4]:
                matrix[term_i][term_j] += tweets_dic[doc_id[0]][4][term_i] * tweets_dic[doc_id[0]][4][term_j]

    matrix_relevant = {}
    words_after_expansion = []
    for word in query_as_list:
        if word in matrix:
            matrix_relevant[word] = []
            for term in matrix[word]:
                if term not in query_as_list:
                    matrix_relevant[word].append(
                        (matrix[word][term] / (matrix[word][word] + matrix[term][term] - matrix[word][term]), term))

    for word in query_as_list:
        if word in matrix_relevant and len(matrix_relevant[word]) > 0:
            words_after_expansion.append(max(matrix_relevant[word], key=lambda item: item[0])[1])

    return words_after_expansion

def word_net_search(query_list):
    """
    The function search for synonyms for each for of the query in order to preform query expansion
    :param query_list: list of terms representing the query
    :return: list of terms representing the query after expansion
    """
    list_to_append = []

    for term in query_list:
        flag = True
        try:
            definition = wn.synset(term + '.n.01').definition()
        except:
            flag = False
        synsets = wn.synsets(term)
        for syn in synsets:
            word = syn.lemmas()[0].name()
            if word != term and flag and definition == syn.definition():
                list_to_append.append(word)
    return query_list + list_to_append


# DO NOT MODIFY CLASS NAME
class Searcher:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit. The 205417637
    # parameter allows you to pass in a precomputed 205417637 that is already in
    # memory for the searcher to use such as LSI, LDA, Word2vec models. 
    # MAKE SURE YOU DON'T LOAD A MODEL INTO MEMORY HERE AS THIS IS RUN AT QUERY TIME.
    def __init__(self, parser, indexer, model=None, **kwargs):
        self._parser = parser
        self._indexer = indexer
        self._model = model
        self.wv = model
        if kwargs.get("word_net"):
            self.word_net_flag = True
        else:
            self.word_net_flag = False
        if kwargs.get("w2v"):
            self.w2v_flag = kwargs.get("w2v")
        else:
            self.w2v_flag = False
        if kwargs.get("tf_idf"):
            self.tf_idf_flag = kwargs.get("tf_idf")
        else:
            self.tf_idf_flag = False
        self._ranker = Ranker(w2v_flag=self.w2v_flag, tf_idf_flag=self.tf_idf_flag)
        if kwargs.get("spell_checker"):
            self.spell_checker_flag = kwargs.get("spell_checker")
        else:
            self.spell_checker_flag = False

    def search_2(self, query):
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
        start = time.time()
        w2v_vector, vector_query, query_as_list = self.initiate_search(query=query)
        ranked_doc_ = self.initiate_ranking(query_as_list, vector_query, w2v_vector)
        ranked_doc_ = ranked_doc_[:500]
        tweets_dict = self._indexer.get_tweets_dict()
        query_as_list += _get_words_for_expansion(ranked_doc_, query_as_list, tweets_dict)
        vector_query = self.get_vector_query(query_as_list)
        w2v_vector = self.get_w2v_vector_query(query_as_list)
        ranked_doc_ = self.initiate_ranking(query_as_list, vector_query, w2v_vector)
        ranked_doc_before = ranked_doc_[:round(len(ranked_doc_) * 0.57)]
        ranked_doc_ids = [doc_id[0] for doc_id in ranked_doc_before]
        print(f"finished searcher in {time.time() - start}")
        return len(ranked_doc_ids), ranked_doc_ids

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
        start = time.time()
        w2v_vector, vector_query, query_as_list = self.initiate_search(query=query)
        ranked_doc_ = self.initiate_ranking(query_as_list, vector_query, w2v_vector)
        ranked_doc_ids = [doc_id[0] for doc_id in ranked_doc_]
        print(f"finished searcher in {time.time() - start}")
        return len(ranked_doc_ids), ranked_doc_ids

    def initiate_ranking(self, query_as_list, vector_query, w2v_vector):
        """
        The function initiates the ranking part
        :param query_as_list: list of terms representing the query
        :param vector_query: vector representing the query based on tf_idf
        :param tweets_dict: dictionary of tweets
        :param w2v_vector: vector representing the query based on w2v
        :return: data frame representing the scores
        """
        tweets_dict = self._indexer.get_tweets_dict()
        relevant_docs, doc_id_set = self._relevant_docs_from_posting(query_as_list)
        ranked_doc_ = Ranker.rank_relevant_docs(relevant_docs,
                                                doc_set=doc_id_set,
                                                vector=np.array(list(vector_query)),
                                                tweets_dict=tweets_dict,
                                                w2v_vector=w2v_vector,
                                                ranker=self._ranker)
        return ranked_doc_

    def initiate_search(self, query):
        """
        This function creates a vector represntation of the query based on tf_idf or w2v
        :param query: string representing the input query
        :return: vectors representing the queries.
        """
        # initiate both list in case we want to use w2v and tf_idf
        vector_query = []
        w2v_vector = []
        query_as_list = self._parser.parse_sentence(query)
        if self.spell_checker_flag:
            query_as_list = spell_checker_search(query_as_list) # spell checker
        if self.word_net_flag:
            query_as_list = word_net_search(query_as_list)
        # query_as_list = self.expand_query(query_as_list, sim_to_expand=0.7)  # expand query based on w2v 205417637
        query_as_list = self._parser.get_lemma_text(query_as_list)
        if self.tf_idf_flag:
            vector_query = self.get_vector_query(query_as_list)
        elif self.w2v_flag:
            w2v_vector = self.get_w2v_vector_query(query_as_list)
        else:
            # basically needs to throw exception.
            w2v_vector = self.get_w2v_vector_query(query_as_list)
        return w2v_vector, vector_query, query_as_list

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
        The function computes a vector representing the query based on tf_idf
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
            try:
                dic_term[key] = value / max_occurrence \
                                * np.log10(num_of_docs / self._indexer.get_term_from_inverted(key))
            except:
                dic_term[key] = value / max_occurrence
        return dic_term.values()

    def get_w2v_vector_query(self, query_list):
        """
        The function compute 300 dimension vector representing the query
        :param query_list: list of terms representing the query
        :return: vector representing the query
        """
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
        """
        The function search for similarity based on w2v for each for of the query in order to preform query expansion
        :param query_as_list: list of terms representing the query
        :param sim_to_expand: choose threshold for similarity of words
        :return: list of terms representing the query after expansion
        """
        new_query_list = []
        for term in query_as_list:
            try:
                list_out = self.wv.most_similar(term)
                new_query_list += [tup[0] for tup in list_out if tup[1] > sim_to_expand]
            except:
                pass
        return query_as_list + new_query_list
