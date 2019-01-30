import re


class BooleanQueryParser:
    def __init__(self, term2termID, docID2doc, termID2docIDs):
        self.operators = ["AND", "OR", "NOT"]
        self.term2termID = term2termID
        self.docID2doc = docID2doc
        self.termID2docIDs = termID2docIDs

    def parse(self, query, not=false):
        query_array = query.split(" ")
        documents = set()
        for word in query_array:
            if word == "AND":

            elif word == "OR":

            else:
                
                set.update(self.termID2docIDs[self.term2termID[word]])
