from program import Corpus, dump, load, parse_file
from matplotlib import pyplot as plt

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
                # index = int(line[3:])
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


def search_and_compare(corpus, query, expected):
    # print("getting results for '{}'".format(query))
    # print("expected :")
    # for docID in expected:
    #     print(corpus.docID2doc[str(docID)])
    # search corpus for this query and get result files
    bulk_results = corpus.search_frequence_tf_idf(query)
    # print("got :")
    # for (docID, _) in bulk_results[:10]:
    #     print(corpus.docID2doc[str(docID)])

    # compare results with expected results
    precisions = []
    rappels = []
    for k in range(1, 21):
        # k is the number of results we return
        result = [docID for (docID, _) in bulk_results[:k]]
        precisions.append(get_precision(result, expected))
        rappels.append(get_rappel(result, expected))
    return precisions, rappels


def compute_f_e_measure(precisions, rappels, k):
    alpha = 0.5  # a modifier
    f_measure = 1/(alpha*precisions[k] + (1-alpha)/rappels[k])
    e_measure = 1 - f_measure
    return f_measure, e_measure


def plot_pression_rappel(precisions, rappels, n):
    x = range(len(rappels))
    # plt.plot(x, rappels, label="rappels" + str(n))
    # plt.plot(x, precisions, label="precisions" + str(n))
    # plt.legend()
    plt.scatter(rappels, precisions, label=n)
    plt.legend()


def get_precision(result, expected):
    # result and expected should be 2 arrays of docIDs, result being of lenght k
    return len(set(result).intersection(set(expected))) / len(result)


def get_rappel(result, expected):
    # result and expected should be 2 arrays of docIDs, result being of lenght k
    return len(set(result).intersection(set(expected))) / len(expected)


if __name__ == "__main__":
    # parse documents and create corpus
    documents = parse_file("Data/CACM/cacm.all")
    corpus = Corpus(documents)

    # load vectors for each file in corpus, and inverted indexes
    corpus.vectors = load('CACM_tf_idf_vectors')
    corpus.term2termID = load("cacm_term2termID")
    corpus.termID2docIDs = load("cacm_termID2docIDs")
    corpus.docID2doc = load("cacm_docID2doc")

    # parse queries and expected queries
    queries = parse_queries('queries/query.text')
    expected_results = parse_expected_results('queries/qrels.text')

    n = 10
    precisions, rappels = search_and_compare(
        corpus, queries[n], expected_results[n])
    #search and compare
    wrong_queries = []

    # for n in range(63):
    for n in [13, 59, 39]:
        precisions, rappels = search_and_compare(
            corpus, queries[n], expected_results[n])
        if(sum(precisions) != 0):
            # plot_pression_rappel(precisions, rappels, n)
            print(compute_f_e_measure(precisions, rappels, len(precisions) - 1))
        else:
            wrong_queries.append(n)
    plt.show()
    print(wrong_queries)
