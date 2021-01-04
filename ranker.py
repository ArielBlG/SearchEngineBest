import time

import numpy as np
import pandas as pd
import utils

# num_of_docs = utils.number_of_docs
num_of_docs = 10000000


class Ranker:
    def __init__(self):
        pass

    @staticmethod
    def get_matrix(relevant_doc, doc_set=None):
        df_matrix = pd.DataFrame()
        se = pd.Series(doc_set)
        df_matrix['Document'] = se.values
        res_dic = {}
        for key in relevant_doc.keys():
            # check np.log 2 or 10
            res_dic[key] = dict(
                map(lambda item: (item[0], item[1] * np.log2(num_of_docs / len(relevant_doc[key]))), relevant_doc[key]))
        relevant_doc.clear()  # in order to ease the ram usage
        for term in res_dic:
            df_matrix[term] = df_matrix['Document'].map(res_dic[term])
        df_matrix = df_matrix.fillna(0)
        return df_matrix

    @staticmethod
    def cos_sim(matrix, vector, tweets_w_ij=None):
        # den = np.sqrt(np.einsum('ij,ij->i',matrix,matrix))*np.einsum('j,j',vector,vector)
        # similarities = 1 - cdist(matrix, vector, metric='cosine')
        res_vector = []
        vector_magnitude = np.sqrt(vector.dot(vector))

        for index, row in enumerate(matrix):
            row_magnitude = np.sqrt(tweets_w_ij[index])
            calc = np.dot(row, vector) / (vector_magnitude * row_magnitude)
            if calc > 1:
                print("a")
            res_vector.append(np.dot(row, vector) / (vector_magnitude * row_magnitude))
        return res_vector
        # den = np.sqrt(np.einsum('ij,ij->i',matrix,matrix)*(vector.dot(vector)))
        # out = matrix.dot(vector) / den
        # return similarities

    @staticmethod
    def rank_relevant_doc(relevant_doc, doc_set=None, vector=None, tweets_dict=None):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param relevant_doc: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """
        matrix_w_id = Ranker.get_matrix(relevant_doc, doc_set)
        start = time.time()
        a = matrix_w_id['Document'].to_list()
        temp = list(map(lambda k: (tweets_dict[k][0], tweets_dict[k][1]), a))
        w_i_j_vector, retweet_vector = zip(*temp)
        end = time.time()
        print("merge data time: ", end - start)
        matrix_wo_id = matrix_w_id.drop(columns=["Document"]).to_numpy()
        start = time.time()
        out = Ranker.cos_sim(matrix=matrix_wo_id, vector=vector,
                             tweets_w_ij=w_i_j_vector)  # .reshape((1,vector.shape[0]))
        end = time.time()
        print("Cossim time: ", end - start)
        retweet_vector = np.array(retweet_vector)
        res_rank = out
        res = list(zip(a, res_rank))
        res.sort(key=lambda item: item[1], reverse=True)
        return res
        # return sorted(relevant_doc.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def retrieve_top_k(sorted_relevant_doc, k=1):
        """
        return a list of top K tweets based on their ranking from highest to lowest
        :param sorted_relevant_doc: list of all candidates docs.
        :param k: Number of top document to return
        :return: list of relevant document
        """
        return sorted_relevant_doc[:k]
