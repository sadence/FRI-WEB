import os
import sys
import unittest

sys.path.append(os.getcwd())
from program import parse_file

class TestParseFile(unittest.TestCase):

    """ Test case for the parse_file method """

    def setUp(self):
        """ Call the parse_file method on the acme.test file """
        print(os.getcwd())
        self.documents = parse_file("test/cacm.test")

    def test_index_number(self):
        """ Test if there is the correct number of documents """
        self.assertEqual(len(self.documents), 39)

    def test_first_doc_properties(self):
        """ Test if the first document contains the correct informations """
        first_doc = self.documents[0]
        self.assertEqual(first_doc.index, 1)
        self.assertEqual(first_doc.title, "Preliminary Report-International Algebraic Language")
        self.assertEqual(first_doc.summary, "")

    def test_tenth_doc_keywords(self):
        """ Test if the first document has the right keywords """
        tenth_doc = self.documents[9]
        self.assertEqual(len(tenth_doc.keywords), 14)
        self.assertEqual(tenth_doc.keywords[0], "standard code")

    def test_multiline_title(self):
        """ Test if a multiline title is correctly interpreted """
        tenth_doc = self.documents[9]
        self.assertEqual(tenth_doc.title, "Code Extension Procedures for Information Interchange* (Proposed USA Standard)")
    
    def test_first_doc_tokens(self):
        """ Test if the first document has the right tokens """
        first_doc = self.documents[0]
        first_doc.tokenize()
        expected_tokens = {
            "Preliminary": 1,
            "Report": 1,
            "International": 1,
            "Algebraic": 1,
            "Language": 1
        }
        self.assertDictEqual(first_doc.tokens, expected_tokens)
    
    def test_tenth_doc_tokens(self):
        """ Test if the tenth document has the right tokens """
        tenth_doc = self.documents[9]
        tenth_doc.tokenize()
        expected_tokens = {
            "Code": 1,
            "Extension": 1,
            "Procedures": 1,
            "for": 1,
            "Information": 1,
            "Interchange": 1,
            "Proposed": 1,
            "USA": 1,
            "Standard": 1,
            "standard": 2,
            "code": 4,
            "information": 1,
            "interchange": 1,
            "characters": 1,
            "shift": 2,
            "out": 1,
            "in": 1,
            "escape": 2,
            "data": 1,
            "link": 1,
            "control": 1,
            "functions":1,
            "procedures": 1,
            "extension": 1,
            "table": 1,
            "bit": 1,
            "pattern": 1
        }
        self.assertDictEqual(tenth_doc.tokens, expected_tokens)

if __name__ == "__main__" :
    unittest.main()
