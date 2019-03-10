
# Parsin queries
with open('./Data/CACM/query.text') as queries_doc:
    queries = []

    markers = {
        'index': '.I',
        'summary': '.W',
        'author': '.A',
        'add_date': '.N'
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
        else:
            if current_marker == 'summary':
                queries[-1] += line

    print(queries[:10])

with open('./Data/CACM/qrels.text') as expected_result_file:
    expected_results = dict()

    for result in expected_result_file:
        res = result.split(" ")
        index = int(res[0]) - 1
        document_id = int(res[1])

        if expected_results.get(index, None) is None:
            expected_results[index] = [document_id]
        else:
            expected_results[index].append(document_id)


if __name__ == "__main__":
