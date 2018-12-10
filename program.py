class Document:

    def __init__(self, index):
        self.index = index
        self.title = ''
        self.keywords = []
        self.summary = ''


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
            'keywords': '.K'
        }

        current_document = None
        current_marker = None

        for line in cacm:
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
            elif line.startswith(markers['summary']):
                current_marker = 'summary'
            elif line.startswith(markers['keywords']):
                current_marker = 'keywords'

            else:
                if current_marker == 'title':
                    current_document.title += line
                elif current_marker == 'summary':
                    current_document.summary += line
                elif current_marker == 'keywords':
                    keywords = line.split(',')
                    current_document.keywords += keywords

        return documents

if __name__ == '__main__':
    documents = parse_file("Data/CACM/cacm.all")
