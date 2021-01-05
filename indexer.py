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
        self.tweets_dic[tweet_id] = [0, document.rt_no_text, document.doc_length]  #
        self.num_of_document += 1
        self.avg_data_size += document.doc_length
        # Go over each term in the doc
        for term in document_dictionary.keys():
            try:
                self.inverted_idx[term] = self.inverted_idx.get(term, 0) + 1
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
        raise NotImplementedError

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, fn):
        """
        Saves a pre-computed index (or indices) so we can save our work.
        Input:
              fn - file name of pickled index.
        """
        raise NotImplementedError

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
