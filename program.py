import re
from matplotlib import pyplot as plt


class Document:

    def __init__(self, index):
        self.index = index
        self.title = ''
        self.keywords = []
        self.summary = ''
        self.tokens = dict()
        self.number_of_tokens = 0

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


class Corpus:

    def __init__(self, documents):
        self.documents = documents
        self.vocabulaire = set()
        self.stop_words = set()
        self.number_of_tokens = 0
        self.frequences = dict()  # does not include stop words

        self.import_stop_words()

        # Construit le vocabulaire
        for document in documents:
            document.tokenize()

            self.vocabulaire.update(document.tokens.keys())
            self.number_of_tokens += document.number_of_tokens
            for token in document.tokens.keys():
                if token not in self.stop_words:
                    self.frequences[token] = self.frequences.get(
                        token, 0) + document.tokens[token]

        self.vocabulaire -= self.stop_words

    def add_document(self, document):
        if isinstance(document, Document):
            self.documents.append(document)
            self.vocabulaire.update(document.tokens.keys())

    def import_stop_words(self):
        with open("./Data/CACM/common_words") as stop_words:
            for line in stop_words:
                self.stop_words.add(line.strip())

    def plot_rank_frequency(self):
        plt.plot(range(len(self.frequences.keys())), sorted(
            self.frequences.values(), reverse=True))
        plt.show()


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
                current_document = Document(index)
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


if __name__ == '__main__':
    documents = parse_file("Data/CACM/cacm.all")

    corpus = Corpus(documents)

    print(corpus.vocabulaire)
    print(len(corpus.vocabulaire))  # sans les stop words
    print(corpus.number_of_tokens)  # avec les stop words

    corpus.plot_rank_frequency()
