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
        not_ = False
        for word in query_array:
            if word == "AND":
                and_ = True
                or_ = False
                not_ = False
            elif word == "OR":
                not_ = False
                and_ = False
                or_ = True
            elif word == "NOT":
                and_ = False
                or_ = False
                not_ = True
            else:
                if and_:
                    documents = documents.intersection(
                        self.termID2docIDs[self.term2termID[word]])
                    and_ = False
                elif or_:
                    documents = documents.union(
                        self.termID2docIDs[self.term2termID[word]])
                    or_ = False
                elif not_:
                    documents = documents.difference(
                        self.termID2docIDs[self.term2termID[word]])
                    or_ = False
                else:
                    documents = documents.union(
                        self.termID2docIDs[self.term2termID[word]])
        return documents
