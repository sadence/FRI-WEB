import re


class BooleanQueryParser:
    def __init__(self, term2termID, docID2doc, termID2docIDs):
        self.operators = ["AND", "OR", "NOT"]
        self.term2termID = term2termID
        self.docID2doc = docID2doc
        self.termID2docIDs = termID2docIDs

    def parse(self, query, not_=False):
        query_array = query.split(" ")
        documents = set()
        and_ = False
        or_ = False
        for word in query_array:
            if word == "AND":
                and_ = True
            elif word == "OR":
                and_ = False
                or_ = True
            else:
                if and_:
                    documents = documents.intersection(
                        self.termID2docIDs[self.term2termID[word]])
                    and_ = False
                elif or_:
                    documents = documents.union(
                        self.termID2docIDs[self.term2termID[word]])
                    or_ = False
                else:
                    documents = documents.union(
                        self.termID2docIDs[self.term2termID[word]])
        return documents
