import re


class Document:

    def __init__(self, index):
        self.index = index
        self.title = ''
        self.keywords = []
        self.summary = ''
        self.tokens = dict()

    def tokenize(self):
        self.tokens = dict()
        reg = re.compile("[\s,\.\{\}\(\)\"-]+")

        # Tokenizing title
        tokenList = reg.split(self.title)
        # Tokenizing summary
        if self.summary != '':
            tokenList += reg.split(self.summary)
        # Tokenizing kewords
        tokenList += self.keywords

        # Counting tokens
        for token in tokenList:
            if token != '':
                self.tokens[token] = self.tokens.get(token, 0) + 1


class Corpus:

    def __init__(self, documents):
        self.documents = documents
        self.vocabulaire = set()

        # Construit le vocabulaire
        for document in documents:
            document.tokenize()

            self.vocabulaire.update(document.tokens.keys())

    def add_document(self, document):
        if isinstance(document, Document):
            self.documents.append(document)
            self.vocabulaire.update(document.tokens.keys())


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
