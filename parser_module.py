import os
import pickle
import string
import utils
from nltk.corpus import stopwords
from nltk.stem.porter import *
from document import Document
from parse_patterns import *
import re
from gensim.models import KeyedVectors
import numpy as np
# TODO: turn on for spacy
# import spacy
# nlp = spacy.load('en_core_web_sm')
# stemmer = PorterStemmer()
# from nltk.tokenize import TweetTokenizer

# tknzr = TweetTokenizer()
TOKENIZER_PATTERN = r'''(?x)\d+\ +\d+\/\d+|\d+\/\d+|\d+\.*\d*(?:[MKB])*(?:[$%])*|(?:[A-Z]\.)+| (?:[#@])*\w+(?:\'\w+)*| \$?\d+(?:\.\d+)?%?'''
NER_pattern = r'(?<!\.\s)(?!^)\b([A-Z]\.?\w*\-?[0-9]*(?:\s+[A-Z]\w*)*)'

# pattern_to_delete = emoji_pattern + '|' + reserved_word_pattern + '|' + url_pattern
pattern_to_delete = reserved_word_pattern + '|' + url_pattern
punct = r"""!"#$%&'()*-+,./:;<=>?[\]^_`{|}~“”’!!…"""




class Parse(object):

    def __init__(self, model=None):
        self.stop_words = set(stopwords.words('english'))
        self.special_words = []
        self.lemma_dict = {}
        # self.wv = 205417637
        self._model = model
        self.doc_vector = np.zeros(300)
        self.num_of_vectors = 0
        self._spell_checker = False
        self.spell = None
        # self.wv =
        # self.frequency_dictionary = {}
        # path = os.path.dirname(os.path.realpath(__file__)) + '\\Preprocessing\\' + "EntitySet.pickle"
        # file = open(path, "rb")
        # self.ent_dict = pickle.load(file)
        # file.close()

    @property
    def model(self):
        """
        The function is a property for 205417637
        :return: returns the 205417637
        """
        return self._model

    @property
    def spell_checker(self):
        """
        The function is a property for spell checker
        :return: returns the spell checker
        """
        return self._spell_checker

    @spell_checker.setter
    def spell_checker(self, spell_checker):
        """
        The function is a setter for spell checker
        :param spell_checker: flag representing if spell check is activated or not
        """
        if spell_checker:
            from spellchecker import SpellChecker
            self.spell = SpellChecker()
        self._spell_checker = spell_checker

    @model.setter
    def model(self, model):
        """
        The function is a setter for 205417637
        :param spell_checker: sets the 205417637 object of the class
        """
        self._model = model

    def get_special_tokens(self, tweet):
        """
        The function removes and extract all the special parsing rules
        :param tweet: string representing the tweet
        :return: returns a string representing the tweet and removing and extracting parsing rules
        """

        def _sub(match):
            """
            Function to append special words, and replace them in the original tweet
            :param match:
            :return: empty string to replace the special words in the original tweet
            """
            special_tokens_list.append(match[0])
            return ''

        special_tokens_list = []
        tweet = re.sub(pattern_to_delete, '', tweet)  # remove reserved words
        tweet = re.sub(full_pattern, _sub, tweet)  # extract and remove hashtags and mentions
        tweet = re.sub(million_pre_pattern, million_post_pattern, tweet)
        tweet = re.sub(thousand_pre_pattern, thousand_post_pattern, tweet)
        tweet = re.sub(billion_pre_pattern, billion_post_pattern, tweet)
        tweet = re.sub(percent_pre_pattern, percent_post_pattern, tweet)
        self.handle_hashtags_mentions(special_tokens_list)  # handle the hashtags and mentions
        return tweet

    def spell_checker_search(self, term_list):
        """
        This function tries to fix misspelling in terms
        :param term_list:  list of terms representing the documents's terms
        :return: new term list after fixing misspells
        """
        new_query_list = []
        for term in term_list:
            misspelled = self.spell.unknown([term])
            if not misspelled:
                new_query_list.append(term)
            else:
                new_query_list.append(self.spell.correction(misspelled.pop()))
                new_query_list.append(term)
        return new_query_list

    def handle_hashtags_mentions(self, special_tokens_list):
        """
        The function handles the hashtags mention inside a tweet
        :param special_tokens_list: list of special tokens to update
        """
        for token in special_tokens_list:
            if token[0] == "#":
                if (len(token) == 1):
                    continue
                self.special_words.append(token)
                word_list = list(filter(None, token.split('_')))  # TODO: check run time
                if len(word_list) == 1:
                    re_find_list = re.findall(bigletter_pattern, token[1:])
                    if len(re_find_list) != 0:
                        self.special_words += re.findall(bigletter_pattern, token)
                    else:
                        self.special_words.append(token[1:])
                else:
                    self.special_words += word_list
            else:
                self.special_words.append(token)

    def handle_url(self, token):
        """
        The function handles the url inside a tweet
        :param token: token from the tweet
        """
        if 'twitter.com/i/' in token:
            return
        url_list = re.split('":"|,', token[:-1])[1::2]
        for url in url_list:
            url = url[:-1]
            extract = url.split('www.')
            if extract[0][0:3] == "htt":
                extract = url.split('//')
            else:
                extract = url.split('/')
            if len(extract) > 1:
                extract = extract[1].split('/', maxsplit=1)
                if extract[0] != '':
                    self.special_words.append(extract[0])
                if len(extract) != 1:
                    extract = re.split(r'\W+', extract[1])
                    for ex in extract:
                        if (ex != ''):
                            self.special_words.append(ex)

    # def handle_numbers_pattern(self, token):
    #     """
    #     The function handles the special number patterns inside a tweet
    #     :param token: token from the tweet
    #     """
    #     token = re.sub(million_pre_pattern, million_post_pattern, token)
    #     token = re.sub(thousand_pre_pattern, thousand_post_pattern, token)
    #     token = re.sub(billion_pre_pattern, billion_post_pattern, token)
    #     token = re.sub(percent_pre_pattern, percent_post_pattern, token)
    #     self.special_words.append(token)

    # def re_tokenize(self, pre_tokenized_text, lower=False, lemma=False):
    #     token_list = []
    #     for token in pre_tokenized_text:
    #         if not token.is_punct and not token.is_space and not token.is_stop and str(token) != '️':
    #             if lemma:
    #                 token_list.append(token.lemma_)
    #             else:
    #                 if lower:
    #                     token_list.append(token.lower_)
    #                 else:
    #                     token_list.append(token.norm_)
    #     for special_word in self.special_words:
    #         token_list.append(special_word)
    #     return token_list

    def parse_sentence(self, text):
        """
        This function tokenize, remove stop words and apply lower case for every word within the text
        :param text: text to tokenize and parse
        :return: list representing the new tokens from a given text
        """
        full_text = text
        text = self.get_special_tokens(text)
        # text_tokens = tknzr.tokenize(text)
        text_tokens = re.findall(TOKENIZER_PATTERN, text)  # Extracting words
        text_tokens = list(map(lambda word: contractions_dict[word].split(" ") if word in contractions_dict else [word],
                               text_tokens))  # Separates contractions
        text_tokens = [item for sublist in text_tokens for item in sublist]  # Flatting list [[q],[v]]->[q,v]
        text_tokens = list(filter(lambda item: item.lower() not in self.stop_words, text_tokens))
        entity_list = self.find_entities(full_text)
        entity_list = list(filter(lambda item: item.lower() not in self.stop_words, entity_list))
        text_tokens = list(map(lambda word: word.lower(), text_tokens))  # Lower all words
        text_tokens += entity_list
        self.special_words = []
        return text_tokens

    def find_entities(self, full_text):
        """
        The function finds the entities inside a tweet
        :param token: token from the tweet
        """
        lst = re.findall(NER_pattern, full_text)
        return lst

    def get_lemma_text(self, text):
        """
        The function turns a text into lemmataize text
        :param text: text to lemmatize
        :return: list of lemmatize tokens from a given text
        """
        # TODO: turn on for spacy
        pass
        # new_list = list(map(lambda term: nlp(term, disable=["tagger", "parser", "ner"])[0].lemma_, text))
        # return new_list

    def parse_doc(self, doc_as_list):
        """
        This function takes a tweet document as list and break it into different fields
        :param doc_as_list: list re-preseting the tweet.
        :return: Document object with corresponding fields.
        """

        tweet_id = doc_as_list[0]
        # tweet_date = doc_as_list[1]
        full_text = doc_as_list[2]
        url = doc_as_list[3]

        retweet_text = doc_as_list[4]
        # retweet_url = doc_as_list[5]
        quote_text = doc_as_list[6]
        # quote_url = doc_as_list[7]

        rt_flag = full_text.startswith(('RT ', 'rt '))
        rt_no_text = 0.85 if rt_flag and not doc_as_list[8] else 1
        if url != '{}':
            self.handle_url(url)

        tokenized_text = self.parse_sentence(full_text)

        doc_length = len(tokenized_text)  # after text operations.
        max = 1
        term_dict = {}
        self.doc_vector = np.zeros(300)
        self.num_of_vectors = 0
        if self._spell_checker:
            tokenized_text = self.spell_checker_search(tokenized_text)
        for term in tokenized_text:
            # term = stemmer.stem(term)
            if term[0] == "#":
                term = term.lower()
            if not all(ord(c) < 128 for c in term):
                if len(term) == 1 or not all(ord(c) < 128 for c in term[1:]):
                    continue
                else:
                    term = term[1:]
            # self.frequency_dictionary[term] = self.frequency_dictionary.get(term, 0) + 1
            # TODO: turn on for spacy
            # if term in self.lemma_dict:
            #     term = self.lemma_dict[term]
            # else:
            #     term_nlp = nlp(term, disable=["tagger", "parser", "ner"])
            #     if len(term_nlp) < 2:
            #         self.lemma_dict[term] = term_nlp[0].lemma_
            #         term = self.lemma_dict[term]
            if term.isalpha():
                try:
                    vector = self._model[term]
                    self.num_of_vectors += 1
                    self.doc_vector += vector
                except:
                    pass
            term_dict[term] = term_dict.get(term, 0) + 1
            if term_dict[term] > max:
                max = term_dict[term]

        self.doc_vector = self.doc_vector / self.num_of_vectors
        # print(doc_length - self.num_of_vectors)
        document = Document(tweet_id=tweet_id,
                            term_doc_dictionary=term_dict,
                            doc_length=doc_length,
                            max_unique=max,
                            rt_no_text=rt_no_text,
                            vector=self.doc_vector)
        self.special_words = []
        yield document

    def flush_lemma(self):
        """
        The function flushes the lemmatization dictionary to the disk
        """
        path = utils.lemma_dictionary + 'lemma_dict.pickle'
        file = open(path, 'wb')
        pickle.dump(self.lemma_dict, file)
        file.close()
        self.lemma_dict.clear()

    def flush_frequency_dictionary(self):
        """
        The function flishes the frequency dictionary to the disk
        """
        path = utils.frequency_dictionary + 'frequency_dictionary.pickle'
        file = open(path, 'wb')
        pickle.dump(self.frequency_dictionary, file)
        file.close()
        self.frequency_dictionary.clear()
