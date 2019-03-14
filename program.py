import json
import math
import pickle
import re
from matplotlib import pyplot as plt
import os
import numpy as np
import math

from query_parser import BooleanQueryParser


def cos_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def test_bqp(term2termID, docID2doc, termID2docIDs):
    bqp = BooleanQueryParser(term2termID, docID2doc, termID2docIDs)
    docIDs_both = bqp.parse("algebra AND math")
    docIDs_algebra = bqp.parse("algebra")
    docIDs_triangles = bqp.parse("math")

    validation_and = True

    for docID in docIDs_both:
        validation_and = validation_and and (
            docID in docIDs_algebra and docID in docIDs_triangles)

    print(f"Testing AND: {validation_and}")

    validation_or = True

    docIDs_both = bqp.parse("algebra OR math")

    for docID in docIDs_triangles:
        validation_or = validation_or and (docID in docIDs_both)
    for docID in docIDs_algebra:
        validation_or = validation_or and (docID in docIDs_both)

    print(f"Testing OR: {validation_or}")

    validation_not = True

    docIDs_both = bqp.parse("algebra NOT math")

    for docID in docIDs_both:
        validation_not = validation_not and (
            docID in docIDs_algebra and docID not in docIDs_triangles)

    print(f"Testing NOT: {validation_not}")


class Document:

    def __init__(self, index=None):
        self.index = index
        self.title = ''
        self.keywords = []
        self.summary = ''
        self.tokens = dict()
        self.number_of_tokens = 0
        self.vector = None

    def tokenize(self):
        # reset in case of retokenization
        self.tokens = dict()
        self.number_of_tokens = 0
        reg = re.compile("[^a-zA-Z0-9]+")  # pylint: disable=W1401

        # Tokenizing title
        tokenList = reg.split(self.title.lower())
        # Tokenizing summary
        if self.summary != '':
            tokenList += reg.split(self.summary.lower())
        # Tokenizing kewords
        for i in range(len(self.keywords)):
            tokenList += reg.split(self.keywords[i].lower())

        # Counting tokens
        for token in tokenList:
            if token != '':
                self.tokens[token] = self.tokens.get(token, 0) + 1
                self.number_of_tokens += 1

    # vector_basis : Corpus.word2index
    def vectorize_frequence_normalisee(self, word2index):
        """Vectorisation par la methode de ponderation Fréquence Normalisée"""
        frequence_max = 0
        # Initialisation du vecteur dans l'espace définit par l'ensemble des mots du corpus
        self.vector = np.array(len(word2index.keys()) * [0])
        for word in word2index.keys():
            # Coordonnée du mot dans l'espace des vecteurs
            index = word2index[word]
            # frequence d'apparition du mot dans le document
            frequence = self.tokens.get(word, 0)
            if frequence_max < frequence:
                frequence_max = frequence
            self.vector[index] = frequence
        self.vector = self.vector / frequence_max

    def vectorize_tf_idf(self, word2index, term2termID, termID2docIDs, len_documents, normalize=False):
        """Vectorisation par la methode de ponderation Fréquence Normalisée"""
        frequence_max = 0
        # Initialisation du vecteur dans l'espace définit par l'ensemble des mots du corpus
        self.vector = np.array(len(word2index.keys()) * [0])
        for word in word2index.keys():
            # Coordonnée du mot dans l'espace des vecteurs
            index = word2index[word]
            # frequence d'apparition du mot dans le document
            tf = self.tokens.get(word, 0)
            idx_corp = term2termID[word]
            idf =  math.log(len_documents / len(termID2docIDs[idx_corp]))
            if normalize and frequence_max < tf:
                frequence_max = tf
            self.vector[index] = tf * idf
        if normalize:
            self.vector = self.vector / frequence_max


class Corpus:

    def __init__(self, documents):
        self.documents = documents
        self.vocabulaire = set()
        self.stop_words = set()
        self.number_of_tokens = 0
        self.frequences = dict()  # does not include stop words
        self.word2index = dict()
        self.vectors = []
        self.term2termID = {}
        self.termID2docIDs = {}
        self.docID2doc = {}

        self.import_stop_words()

        # Construit le vocabulaire
        for document in documents:
            document.tokenize()
            self.update_corpus(document)

        self.vocabulaire -= self.stop_words

        self.build_word2index()

    def add_document(self, document):
        if isinstance(document, Document):
            self.documents.append(document)
            self.update_corpus(document)

    def update_corpus(self, document):
        self.vocabulaire.update(document.tokens.keys())
        self.number_of_tokens += document.number_of_tokens
        self.update_frequence(document)

    def update_frequence(self, document):
        for token in document.tokens.keys():
            if token not in self.stop_words:
                self.frequences[token] = self.frequences.get(
                    token, 0) + document.tokens[token]

    def import_stop_words(self):
        with open("./Data/CACM/common_words") as stop_words:
            for line in stop_words:
                self.stop_words.add(line.strip())

    def plot_rank_frequency(self):
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.frequences.keys())), sorted(
            self.frequences.values(), reverse=True))
        plt.title("Graphe frequence / rang")

        plt.subplot(2, 1, 2)
        plt.plot([math.log(f) for f in range(1, len(self.frequences.keys()) + 1)], [math.log(r) for r in sorted(
            self.frequences.values(), reverse=True)])
        plt.title("Graphe log(f) / log(r)")

        plt.show()

    def build_word2index(self):
        vocabulaire_list = list(self.vocabulaire)
        vocabulaire_list.sort()

        for i in range(len(vocabulaire_list)):
            self.word2index[vocabulaire_list[i]] = i

    def vectorize_documents(self, method="freq_norm", normalize=False):
        if method == "tf_idf":
            if not self.term2termID or not self.termID2docIDs:
                raise Exception("Missing inverted index")
            for document in self.documents:
                document.vectorize_tf_idf(self.word2index, self.term2termID, self.termID2docIDs, len(self.documents), normalize=normalize)
        else:
            for document in self.documents:
                document.vectorize_frequence_normalisee(self.word2index)
        self.vectors = [document.vector for document in corpus.documents]


    def search_frequence_normalisee(self, query):
        if len(self.vectors) == 0:
            raise Exception("Cannot search Corpus. Documents not vectorized.")

        frequence_max = 0
        vocab_size = len(self.word2index.keys())
        reg = re.compile("[^a-zA-Z0-9]+")
        query_vector = np.array(vocab_size * [0])

        # Vectorisation
        query = reg.split(query.lower())
        for term in query:
            index = self.word2index.get(term, None)
            if not index is None:
                query_vector[index] += 1

        frequence_max = max(query_vector)
        if frequence_max == 0:
            raise Exception("No match found.")
        query_vector = query_vector / frequence_max

        # Cosine
        cossims = [(i, cos_sim(query_vector, vector))
                   for i, vector in enumerate(self.vectors)]

        return sorted(cossims, key=lambda tup: -tup[1])

    def search_frequence_tf_idf(self, query, normalize=False):
        if len(self.vectors) == 0:
            raise Exception("Cannot search Corpus. Documents not vectorized.")

        vocab_size = len(self.word2index.keys())
        reg = re.compile("[^a-zA-Z0-9]+")
        query_vector = np.array(vocab_size * [0])
        # Vectorisation
        query = reg.split(query.lower())
        for term in query:
            index = self.word2index.get(term, None)
            if not index is None:
                query_vector[index] += 1 # computes tf
        frequence_max = max(query_vector)
        # compute idf when tf computation is done
        for term in query:
            index = self.word2index.get(term, None)
            if not index is None:
                idf = math.log(len(self.documents) / len(self.termID2docIDs[self.term2termID[term]]))
                query_vector[index] *= idf
        if normalize:
            query_vector = query_vector / frequence_max



        # Cosine
        cossims = [(i, cos_sim(query_vector, vector))
                   for i, vector in enumerate(self.vectors)]

        return sorted(cossims, key=lambda tup: -tup[1])


def parse_file(file_location):

    with open(file_location, 'r') as cacm:

        documents = []

        markers = {
            'index': '.I',
            'title': '.T',
            'summary': '.W',
            'publication_date': '.B',
            'author': '.A',
            'add_date': '.N',
            'references': '.X',
            'keywords': '.K',
            'chapters': '.C'
        }

        current_document = None
        current_marker = None

        for line in cacm:
            line = line.strip()
            if line is None:
                pass
            elif line.startswith(markers['index']):
                index = int(line[3:])
                current_document = Document(index=index)
                documents.append(current_document)
            elif line.startswith(markers['title']):
                current_marker = 'title'
            elif line.startswith(markers['publication_date']):
                # ignore marker
                current_marker = None
            elif line.startswith(markers['author']):
                # ignore marker
                current_marker = None
            elif line.startswith(markers['add_date']):
                # ignore marker
                current_marker = None
            elif line.startswith(markers['references']):
                # ignore marker
                current_marker = None
            elif line.startswith(markers['chapters']):
                # ignore marker
                current_marker = None
            elif line.startswith(markers['summary']):
                current_marker = 'summary'
            elif line.startswith(markers['keywords']):
                current_marker = 'keywords'

            else:
                line = line.lower()
                if current_marker == 'title':
                    if len(current_document.title) > 0:
                        current_document.title += " "
                    current_document.title += line
                elif current_marker == 'summary':
                    if len(current_document.summary) > 0:
                        current_document.summary += " "
                    current_document.summary += line
                elif current_marker == 'keywords':
                    keywords = line.split(',')
                    current_document.keywords += [k.strip()
                                                  for k in keywords if k != ""]

        return documents


def recursive_read_files(directory_path, blocks=False):
    """[summary]

    Arguments:
        directory_path {[type]} -- [description]

    Keyword Arguments:
        blocks {bool} -- Return a list of blocks of tokenized documents (default: {False})
    """

    directories = os.listdir(directory_path)
    directories = [
        directory for directory in directories if directory[0] != '.']
    print(directories)
    for dir_name in directories:
        if blocks:
            block = []
        files = os.listdir(directory_path + dir_name + '/')
        for file_name in files:
            with open(directory_path + dir_name + '/' + file_name, 'r') as file:
                current_document = Document()
                current_document.title = "{}/{}".format(dir_name, file_name)
                current_document.summary = file.read()
                if blocks:
                    # normally corpus calls tokenize
                    current_document.tokenize()
                    block.append(current_document)
                else:
                    yield current_document
        if blocks:
            yield block
        print("Done directory {}".format(dir_name))


def create_inverted_index(blocks, stop_words):
    # blocks is a list of iterators, with number_blocks = len(blocks)
    # each block is a list of documents
    # on considerera apres le fait de faire un BSTree
    term2termID = dict()
    docID2doc = dict()
    termID2docIDs = dict()
    for block in blocks:
        for document in block:
            docID = str(len(docID2doc))
            docID2doc[docID] = document.title
            for token in document.tokens.keys():
                if token not in stop_words:
                    termID = term2termID.get(token)
                    if termID is None:
                        term2termID[token] = str(len(term2termID))
                        termID = term2termID[token]
                    if termID2docIDs.get(termID) is None:
                        termID2docIDs[termID] = []
                    termID2docIDs[termID].append(docID)
    return term2termID, docID2doc, termID2docIDs


def dump(file_name, data, use="pickle"):
    if use == "json":
        with open(file_name + ".json", 'w') as outputfile:
            json.dump(data, outputfile)
    else:
        with open(file_name + ".pickle", "wb") as outputfile:
            pickle.dump(data, outputfile)


def load(file_name, use="pickle"):
    if use == "json":
        with open(file_name + ".json", 'r') as inputfile:
            data = json.load(inputfile)
    else:
        with open(file_name + ".pickle", "rb") as inputfile:
            data = pickle.load(inputfile)
    return data


if __name__ == '__main__':
    documents = parse_file("Data/CACM/cacm.all")

    corpus = Corpus(documents)

    # print(corpus.vocabulaire)
    print("vocabulaire", len(corpus.vocabulaire))  # sans les stop words
    print("tokens", corpus.number_of_tokens)  # avec les stop words

    # # corpus.plot_rank_frequency()

    term2termID, docID2doc, termID2docIDs = create_inverted_index(
         [corpus.documents], corpus.stop_words)
    corpus.term2termID = term2termID
    corpus.docID2doc = docID2doc
    corpus.termID2docIDs = termID2docIDs

    print(len(corpus.term2termID), len(corpus.word2index))

    # word = "algebra"
    # wordID = term2termID.get(word)
    # documentsID = termID2docIDs.get(wordID)
    # documents_titles = [docID2doc.get(documentID)
    #                     for documentID in documentsID]
    # print(documents_titles)

    # term2termID, docID2doc, termID2docIDs = create_inverted_index(
    #     [corpus.documents], corpus.stop_words)
    # word = "algebra"
    # wordID = term2termID.get(word)
    # documentsID = termID2docIDs.get(wordID)
    # documents_titles = [docID2doc.get(documentID)
    #                     for documentID in documentsID]
    dump('cacm_term2termID', term2termID)
    dump('cacm_docID2doc', docID2doc)
    dump('cacm_termID2docIDs', termID2docIDs)
    # data = load('cacm_docID2doc')
    # print(data["1"])
    # print(documents_titles)

    # listeuh = recursive_read_files('./Data/CS276/pa1-data/')
    # corpus = Corpus(listeuh)

    # corpus = Corpus([])
    # stop_words_set = set()
    # with open("./Data/CACM/common_words") as stop_words:
    #     for line in stop_words:
    #         stop_words_set.add(line.strip())
    #     blocks = recursive_read_files('./Data/CS276/pa1-data/', blocks=True)

    #     term2termID, docID2doc, termID2docIDs = create_inverted_index(
    #         blocks, stop_words_set)
    #     word = "algebra"
    # wordID = term2termID.get(word)
    # documentsID = termID2docIDs.get(wordID)
    # documents_titles = [docID2doc.get(documentID)
    #                     for documentID in documentsID]
    # print(documents_titles)

    # term2termID = load('term2termID')
    # docID2doc = load('docID2doc')
    # termID2docIDs = load('termID2docIDs')

    # test_bqp(term2termID, docID2doc, termID2docIDs)

    # --

    # for block in blocks:
    #     for document in block:
    #         corpus.add_document(document)
    # print(len(corpus.vocabulaire))
    # print(corpus.number_of_tokens)
    # corpus.plot_rank_frequency()

    # DUMPING VECTORS
    # corpus.vectorize_documents(method="tf_idf")

    # data = [document.vector for document in corpus.documents]

    # dump('CACM_tf_idf_vectors', data)

    # END DUMPING VECTORS

    # corpus.vectors = load('CACM_tf_idf_vectors')
    # corpus.term2termID = term2termID
    # corpus.docID2doc = docID2doc
    # corpus.termID2docIDs = termID2docIDs

    # query = "A Routine to Find the Solution of Simultaneous Linear Equations with Polynomial Coefficients"
    # result = corpus.search_frequence_tf_idf(query)[:20]
    # human_readable = [(corpus.documents[idx].title, cos)
    #                   for idx, cos in result]

    # print(human_readable)
