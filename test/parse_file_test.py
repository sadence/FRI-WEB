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

if __name__ == "__main__" :
    unittest.main()
