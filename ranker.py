import time

import numpy as np
import pandas as pd
from numpy.linalg import norm
import utils


# num_of_docs = utils.number_of_docs
# num_of_docs = 10000000


class Ranker:
    def __init__(self, w2v_flag=True, tf_idf_flag=False):
        self._w2v_flag = w2v_flag
        self._tf_idf_flag = tf_idf_flag

    def __getattr__(self, item):
        pass

    @property
    def w2v_flag(self):
        """
        The function is a getter for the w2v flag
        :return: w2v_flag
        """
        return self._w2v_flag

    @property
    def tf_idf_flag(self):
        """
        The function is a getter for the tf_idf flag
        :return: tf_idf_flag
        """
        return self._tf_idf_flag

    @staticmethod
    def get_matrix(relevant_doc, doc_set=None, num_of_docs=10000000):
        """
        :param relevant_doc: dictionary of relevant documents based on the query
        :param doc_set: set of all documents ids in the corpus
        :param num_of_docs: number of documents in the corpus
        :return: numpy matrix containing the vector of each documents based on tf_idf computation
        """
        df_matrix = pd.DataFrame()
        se = pd.Series(list(doc_set))
        df_matrix['Document'] = se.values
        res_dic = {}
        for key in relevant_doc.keys():
            # check np.log 2 or 10
            res_dic[key] = dict(
                map(lambda item: (item[0], item[1] * np.log10(num_of_docs / len(relevant_doc[key]))),
                    relevant_doc[key]))
        relevant_doc.clear()  # in order to ease the ram usage
        for term in res_dic:
            df_matrix[term] = df_matrix['Document'].map(res_dic[term])
        df_matrix = df_matrix.fillna(0)
        return df_matrix

    # df_matrix.loc[df_matrix['Document'] == 1287320400620335104]
    @staticmethod
    def cos_sim(matrix, vector, tweets_w_ij=None, w2v_matrix=None, w2v_vector=None, weights_w2v=0.5):
        """
        :parm matrix: numpy matrix containing the vector of each documents based on tf_idf computation
        :parm vector:  numpy vector containing the vector of query based on tf_idf computation
        :parm tweets_w_ij: numpy matrix containing the magnitude of each document based on tf_idf computation
        :return:
        """
        res_vector = []
        vector_magnitude = np.sqrt(vector.dot(vector))
        norm_w2v_vector = norm(w2v_vector)
        for index, row in enumerate(matrix):
            row_magnitude = np.sqrt(tweets_w_ij[index])
            cos_sim_tf_idf = np.dot(row, vector) / (vector_magnitude * row_magnitude)
            cos_sim_w2v = np.dot(w2v_vector, w2v_matrix[index]) / (norm_w2v_vector * norm(w2v_matrix[index]))

            res_vector.append((1-weights_w2v) * cos_sim_tf_idf + weights_w2v*cos_sim_w2v)
            # res_vector.append(cos_sim_w2v)
        return res_vector

    @staticmethod
    def cos_sim_w2v(matrix, w2v_matrix=None, w2v_vector=None):
        """
        :parm matrix: numpy matrix containing the vector of each documents based on w2v computation
        :parm vector:  numpy vector containing the vector of query based on w2v computation
        :parm tweets_w_ij: numpy matrix containing the magnitude of each document based on w2v computation
        :return:
        """
        res_vector = []
        norm_w2v_vector = norm(w2v_vector)
        for index, row in enumerate(matrix):
            cos_sim_w2v = np.dot(w2v_vector, w2v_matrix[index]) / (norm_w2v_vector * norm(w2v_matrix[index]))
            res_vector.append(cos_sim_w2v)
        return res_vector

    @staticmethod
    def cos_sim_tf_idf(matrix, vector=None, tweets_w_ij=None):
        """
        :parm matrix: numpy matrix containing the vector of each documents based on tf_idf computation
        :parm w2v_vector:  numpy vector containing the vector of query based on tf_idf computation
        :parm tweets_w_ij: numpy matrix containing the magnitude of each document based on tf_idf computation
        :return:
        """
        res_vector = []
        vector_magnitude = np.sqrt(vector.dot(vector))
        for index, row in enumerate(matrix):
            row_magnitude = np.sqrt(tweets_w_ij[index])
            cos_sim_tf_idf = np.dot(row, vector) / (vector_magnitude * row_magnitude)
            res_vector.append(cos_sim_tf_idf)
        return res_vector

    @staticmethod
    def rank_relevant_docs(relevant_doc, doc_set=None, vector=None, tweets_dict=None, w2v_vector=None, ranker=None):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param relevant_doc: dictionary of documents that contains at least one term from the query.
        :param doc_set:
        :param vector:
        :param tweets_dict:
        :param w2v_vector:
        :return: sorted list of documents by score
        """
        matrix_w_id = Ranker.get_matrix(relevant_doc, doc_set, num_of_docs=len(tweets_dict))
        # start = time.time()
        a = matrix_w_id['Document'].values.tolist()
        if ranker.tf_idf_flag and ranker.w2v_flag:
            out = Ranker.rank_relevant_docs_both(tweets_dict=tweets_dict,
                                                 lst_matrix_w_id=a,
                                                 vector=vector,
                                                 w2v_vector=w2v_vector,
                                                 matrix_w_id=matrix_w_id)
        elif ranker.w2v_flag:
            out = Ranker.rank_relevant_docs_w2v(tweets_dict=tweets_dict, lst_matrix_w_id=a,
                                                w2v_vector=w2v_vector, matrix_w_id=matrix_w_id)
        else:
            out = Ranker.rank_relevant_docs_tf_idf(tweets_dict=tweets_dict, lst_matrix_w_id=a,
                                                   vector=vector, matrix_w_id=matrix_w_id)

        # temp = list(map(lambda k: (tweets_dict[k][0], tweets_dict[k][1], tweets_dict[k][3]), a))
        # w_i_j_vector, retweet_vector, w2v_matrix = zip(*temp)
        # end = time.time()
        # print("merge data time: ", end - start)
        # matrix_wo_id = matrix_w_id.drop(columns=["Document"]).to_numpy()
        # start = time.time()
        # if ranker.w2v_flag():
        #     out = Ranker.cos_sim_w2v(matrix=matrix_wo_id, w2v_matrix=w2v_matrix, w2v_vector=w2v_vector)
        # out = Ranker.cos_sim(matrix=matrix_wo_id, vector=vector,
        #                      tweets_w_ij=w_i_j_vector, w2v_matrix=w2v_matrix,
        #                      w2v_vector=w2v_vector)  # .reshape((1,vector.shape[0]))
        # if ranker.tf_idf_flag():
        #     out = Ranker.cos_sim_tf_idf(matrix=matrix_wo_id, vector=vector, tweets_w_ij=w_i_j_vector)
        # end = time.time()
        # print("Cossim time: ", end - start)
        # retweet_vector = np.array(retweet_vector)
        res_rank = out
        res = list(zip(a, res_rank))
        res.sort(key=lambda item: item[1], reverse=True)
        return res
        # return sorted(relevant_doc.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def rank_relevant_docs_w2v(tweets_dict=None, lst_matrix_w_id=None, w2v_vector=None, matrix_w_id=None):
        """
        This function provides rank for each relevant document based on w2v representation.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param tweets_dict: tweets dic representing all the tweets
        :param lst_matrix_w_id: list of matrix ids
        :param w2v_vector: vector representation of the query
        :param matrix_w_id: dataframe for matrix of all needed data
        :return: sorted list of documents by score
        """
        temp = list(map(lambda k: (tweets_dict[k][1], tweets_dict[k][3]), lst_matrix_w_id))
        retweet_vector, w2v_matrix = zip(*temp)
        # matrix_wo_id = matrix_w_id.drop(columns=["Document"]).to_numpy()
        matrix_w_id = matrix_w_id.drop(columns=["Document"])
        matrix_wo_id = np.array(matrix_w_id)

        return Ranker.cos_sim_w2v(matrix=matrix_wo_id, w2v_matrix=w2v_matrix, w2v_vector=w2v_vector)

    @staticmethod
    def rank_relevant_docs_tf_idf(tweets_dict=None, lst_matrix_w_id=None, vector=None, matrix_w_id=None):
        """
        This function provides rank for each relevant document based on tf_idf representation.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param tweets_dict: tweets dic representing all the tweets
        :param lst_matrix_w_id: list of matrix ids
        :param vector: vector representation of the query
        :param matrix_w_id: dataframe for matrix of all needed data
        :return: sorted list of documents by score
        """
        temp = list(map(lambda k: (tweets_dict[k][0], tweets_dict[k][1]), lst_matrix_w_id))
        w_i_j_vector, retweet_vector = zip(*temp)
        matrix_w_id = matrix_w_id.drop(columns=["Document"])
        matrix_wo_id = np.array(matrix_w_id)
        # matrix_wo_id = matrix_w_id.drop(columns=["Document"]).to_numpy()

        return Ranker.cos_sim_tf_idf(matrix=matrix_wo_id, vector=vector, tweets_w_ij=w_i_j_vector)

    @staticmethod
    def rank_relevant_docs_both(tweets_dict=None, lst_matrix_w_id=None, vector=None, w2v_vector=None, matrix_w_id=None):
        """
        This function provides rank for each relevant document based on w2v & tf_idf representation.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param tweets_dict: tweets dic representing all the tweets
        :param lst_matrix_w_id: list of matrix ids
        :param vector: vector representation of the query by tf-idf representation
        :param w2v_vector: vector representation of the query by w2v representation
        :param matrix_w_id: dataframe for matrix of all needed data
        :return: sorted list of documents by score
        """
        temp = list(map(lambda k: (tweets_dict[k][0], tweets_dict[k][1], tweets_dict[k][3]), lst_matrix_w_id))
        w_i_j_vector, retweet_vector, w2v_matrix = zip(*temp)
        # matrix_wo_id = matrix_w_id.drop(columns=["Document"]).to_numpy()
        matrix_w_id = matrix_w_id.drop(columns=["Document"])
        matrix_wo_id = np.array(matrix_w_id)
        return Ranker.cos_sim(matrix=matrix_wo_id, vector=vector,
                              tweets_w_ij=w_i_j_vector,
                              w2v_matrix=w2v_matrix,
                              w2v_vector=w2v_vector,
                              weights_w2v=0.5)

    @staticmethod
    def retrieve_top_k(sorted_relevant_doc, k=1):
        """
        return a list of top K tweets based on their ranking from highest to lowest
        :param sorted_relevant_doc: list of all candidates docs.
        :param k: Number of top document to return
        :return: list of relevant document
        """
        return sorted_relevant_doc[:k]
