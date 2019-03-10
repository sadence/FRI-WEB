from program import Corpus, dump, load, parse_file

# Parsin queries
def parse_queries(path):
    with open(path) as queries_doc:
        queries = []

        markers = {
            'index': '.I',
            'summary': '.W',
            'author': '.A',
            'add_date': '.N',
            'query': '.V'
        }

        current_marker = None

        for line in queries_doc:
            line = line.strip()
            if line is None:
                pass
            elif line.startswith(markers['index']):
                index = int(line[3:])
                queries.append("")
                current_marker = 'index'
            elif line.startswith(markers['summary']):
                current_marker = 'summary'
            elif line.startswith(markers['author']):
                current_marker = 'author'
            elif line.startswith(markers['add_date']):
                current_marker = 'add_date'
            elif line.startswith(markers['query']):
                current_marker = 'query'
            else:
                if current_marker == 'query':
                    queries[-1] += line

        return queries

def parse_expected_results(path):
    with open(path) as expected_result_file:
        expected_results = []

        for result in expected_result_file:
            res = result.split(" ")
            index = int(res[0]) - 1
            document_id = int(res[1])

            if len(expected_results) < index + 1:
                expected_results.append([document_id])
            else:
                expected_results[index].append(document_id)
    return expected_results

def search_and_compare(query, expected):
    print("this is a test")


if __name__ == "__main__":
    # parse documents and create corpus
    documents = parse_file("Data/CACM/cacm.all")
    corpus = Corpus(documents)

    # load vectors for each file in corpus, and inverted indexes
    corpus.vectors = load('CACM_tf_idf_vectors')
    term2termID = load("term2termID")
    termID2docIDs = load("termID2docIDs")

    # parse queries and expected queries
    queries = parse_queries('queries/query.text')
    expected_results = parse_expected_results('queries/qrels.text')

    # search
    query = "A Routine to Find the Solution of Simultaneous Linear Equations with Polynomial Coefficients"
    result = corpus.search_frequence_tf_idf(query, term2termID, termID2docIDs, normalize=True)[:20]
    human_readable = [(corpus.documents[idx].title, cos)
                      for idx, cos in result]
    
    print(human_readable)