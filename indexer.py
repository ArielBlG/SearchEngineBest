import numpy as np
import utils


# DO NOT MODIFY CLASS NAME
class Indexer:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def __init__(self, config):
        self.inverted_idx = {}
        self.postingDict = {}
        self.config = config
        self.tweets_dic = {}
        self.num_of_document = 0
        self.avg_data_size = 0

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def add_new_doc(self, document):
        """
        This function perform indexing process for a document object.
        Saved information is captures via two dictionaries ('inverted index' and 'posting')
        :param document: a document need to be indexed.
        :return: -
        """

        document_dictionary = document.term_doc_dictionary
        max_unique = document.max_unique
        tweet_id = document.tweet_id
        self.tweets_dic[tweet_id] = [0, document.rt_no_text, document.doc_length, document.doc_vector]  #
        self.num_of_document += 1
        self.avg_data_size += document.doc_length
        # doc_vector = document.doc_vector
        # Go over each term in the doc
        for term in document_dictionary.keys():
            try:
                self.inverted_idx[term] = self.inverted_idx.get(term, 0)
                self.inverted_idx[term] += 1
                doc_dict_term = document_dictionary[term]
                if term in self.postingDict:
                    self.postingDict[term].append((tweet_id, doc_dict_term / max_unique, doc_dict_term))
                else:
                    self.postingDict[term] = [(tweet_id, doc_dict_term / max_unique, doc_dict_term)]

            except Exception as e:
                e.with_traceback()
                print('problem with the following key {}'.format(term[0]))

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        object_to_open = utils.load_obj(fn)
        self.inverted_idx = object_to_open[0]
        self.postingDict = object_to_open[1]
        self.tweets_dic = object_to_open[2]

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, fn):
        """
        Saves a pre-computed index (or indices) so we can save our work.
        Input:
              fn - file name of pickled index.
        """
        object_to_save = (self.inverted_idx, self.postingDict, self.tweets_dic)
        utils.save_obj(object_to_save, fn)

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def _is_term_exist(self, term):
        """
        Checks if a term exist in the dictionary.
        """
        return term in self.postingDict

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def get_term_posting_list(self, term):
        """
        Return the posting list from the index for a term.
        """
        return self.postingDict[term] if self._is_term_exist(term) else []

    def get_term_from_inverted(self, term):
        """
        retur the value of the received term in the inverted index
        """
        return self.inverted_idx[term]

    def get_number_of_docs(self):
        """
        return the number of documents in the corpus
        """
        return len(self.tweets_dic)

    # TODO: need to be changed
    def get_tweets_dict(self):
        """

        """
        return self.tweets_dic

    def compute_weights_per_doc(self):
        """
        Updates W_ij for each document
        """
        terms_to_remove = []
        for term, posting_list in self.postingDict.items():
            if len(self.postingDict[term]) == 1:
                terms_to_remove.append(term)
            for tweet in posting_list:
                self.tweets_dic[tweet[0]][0] += (tweet[1] *
                                                 np.log10(self.num_of_document / self.inverted_idx[term])) ** 2
        # for term in terms_to_remove:
        #     self.postingDict.pop(term)
        # print(len(self.tweets_dic.keys()))
        # print(self.num_of_document)
