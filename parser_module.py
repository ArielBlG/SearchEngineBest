import os
import pickle
import string
import utils
from nltk.corpus import stopwords

from document import Document
from parse_patterns import *
import re
import spacy

# from nltk.tokenize import TweetTokenizer

# tknzr = TweetTokenizer()
TOKENIZER_PATTERN = r'''(?x)\d+\ +\d+\/\d+|\d+\/\d+|\d+\.*\d*(?:[MKB])*(?:[$%])*|(?:[A-Z]\.)+| (?:[#@])*\w+(?:\'\w+)*| \$?\d+(?:\.\d+)?%?'''
NER_pattern = r'(?<!\.\s)(?!^)\b([A-Z]\.?\w*\-?[0-9]*(?:\s+[A-Z]\w*)*)'
nlp = spacy.load('en_core_web_sm')
# pattern_to_delete = emoji_pattern + '|' + reserved_word_pattern + '|' + url_pattern
pattern_to_delete = reserved_word_pattern + '|' + url_pattern


class Parse:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.special_words = []
        self.lemma_dict = {}
        # self.frequency_dictionary = {}
        # path = os.path.dirname(os.path.realpath(__file__)) + '\\Preprocessing\\' + "EntitySet.pickle"
        # file = open(path, "rb")
        # self.ent_dict = pickle.load(file)
        # file.close()

    def get_special_tokens(self, tweet):
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

    def handle_hashtags_mentions(self, special_tokens_list):
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
        if 'twitter.com/i/web' in token:
            return
        url_list = re.split('":"|,', token[:-1])[1::2]
        for url in url_list:
            url = url[:-1]
            extract = url.split('www.')
            if len(extract) == 1:
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

    def handle_ents(self, token):
        token = re.sub(million_pre_pattern, million_post_pattern, token)
        token = re.sub(thousand_pre_pattern, thousand_post_pattern, token)
        token = re.sub(billion_pre_pattern, billion_post_pattern, token)
        token = re.sub(percent_pre_pattern, percent_post_pattern, token)
        self.special_words.append(token)

    def re_tokenize(self, pre_tokenized_text, lower=False, lemma=False):
        token_list = []
        for token in pre_tokenized_text:
            if not token.is_punct and not token.is_space and not token.is_stop and str(token) != '️':
                if lemma:
                    token_list.append(token.lemma_)
                else:
                    if lower:
                        token_list.append(token.lower_)
                    else:
                        token_list.append(token.norm_)
        for special_word in self.special_words:
            token_list.append(special_word)
        return token_list

    def parse_sentence(self, text):
        """
        This function tokenize, remove stop words and apply lower case for every word within the text
        :param text:
        :return:
        """
        full_text = text
        text = self.get_special_tokens(text)
        # text_tokens = tknzr.tokenize(text)
        text_tokens = re.findall(TOKENIZER_PATTERN, text)
        new_text_tokens = [contractions_dict[word.lower()] if word.lower() in contractions_dict else word for word in
                           text_tokens]
        punct = string.punctuation + "“”’!!…"
        punct.replace('@', '')
        text_tokens_without_stopwords = [w.lower() for w in new_text_tokens if
                                         w.lower() not in self.stop_words and w not in punct]
        # text_tokens_without_stopwords = [w for w in new_text_tokens if w.lower() not in self.stop_words and w not in punct]
        # text_tokens_without_stopwords = [w.lower() for w in text_tokens if w.lower() not in self.stop_words]
        entity_list = []
        entity_list = self.find_ent(full_text)
        new_text_tokens = [contractions_dict[word.lower()] if word.lower() in contractions_dict else word for word in
                           entity_list]
        entity_list = [w for w in entity_list if w.lower() not in self.stop_words]
        new_text_tokens.clear()
        text_tokens_without_stopwords += [word for word in entity_list if word not in text_tokens_without_stopwords]
        text_tokens_without_stopwords += [token for token in self.special_words]
        return text_tokens_without_stopwords

    def find_ent(self, full_text):
        lst = re.findall(NER_pattern, full_text)
        return lst

    def get_lemma_text(self, text):
        new_list = list(map(lambda term: nlp(term, disable=["tagger", "parser", "ner"])[0].lemma_, text))
        return new_list

    def parse_doc(self, doc_as_list):
        """
        This function takes a tweet document as list and break it into different fields
        :param doc_as_list: list re-preseting the tweet.
        :return: Document object with corresponding fields.
        """
        self.special_words = []
        tweet_id = doc_as_list[0]
        # tweet_date = doc_as_list[1]
        full_text = doc_as_list[2]
        url = doc_as_list[3]

        retweet_text = doc_as_list[4]
        # retweet_url = doc_as_list[5]
        quote_text = doc_as_list[6]
        # quote_url = doc_as_list[7]

        rt_flag = full_text.startswith(('RT ','rt '))
        rt_no_text = 0.85 if rt_flag and not doc_as_list[8] else 1
        if url != '{}' and not rt_no_text:
            self.handle_url(url)

        tokenized_text = self.parse_sentence(full_text)

        doc_length = len(tokenized_text)  # after text operations.
        max = 1
        term_dict = {}
        for term in tokenized_text:
            if term[0] == "#":
                term = term.lower()
            if not all(ord(c) < 128 for c in term):
                if len(term) == 1 or not all(ord(c) < 128 for c in term[1:]):
                    continue
                else:
                    term = term[1:]
            # self.frequency_dictionary[term] = self.frequency_dictionary.get(term, 0) + 1

            if term in self.lemma_dict:
                term = self.lemma_dict[term]
            else:
                term_nlp = nlp(term, disable=["tagger", "parser", "ner"])
                if len(term_nlp) < 2:
                    self.lemma_dict[term] = term_nlp[0].lemma_
                    term = self.lemma_dict[term]
            if term not in term_dict.keys():
                term_dict[term] = 1
            else:
                term_dict[term] += 1
                if term_dict[term] > max:
                    max = term_dict[term]

        document = Document(tweet_id=tweet_id, term_doc_dictionary=term_dict, doc_length=doc_length, max_unique=max,rt_no_text=rt_no_text)
        return document

    def flush_lemma(self):
        path = utils.lemma_dictionary + 'lemma_dict.pickle'
        file = open(path, 'wb')
        pickle.dump(self.lemma_dict, file)
        file.close()
        self.lemma_dict.clear()

    def flush_frequency_dictionary(self):
        path = utils.frequency_dictionary + 'frequency_dictionary.pickle'
        file = open(path, 'wb')
        pickle.dump(self.frequency_dictionary, file)
        file.close()
        self.frequency_dictionary.clear()
