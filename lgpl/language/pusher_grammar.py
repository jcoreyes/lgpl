import random

import nltk
from nltk.parse.generate import generate


class HighLevelGrammar:
	"""
	Place in skinny rectangle in front of you
	Place in square behind you
	"""
	def __init__(self):
		VERBS = ['"Place objects in"']
		ADJ = ['"skinny rectangle"', '"fat rectangle"', '"square"']
		N = ['"formation"']
		PREP = ['"in front of"', '"behind"']
		YOU = ['"you"']
		self.grammar = nltk.CFG.fromstring("""
        S -> V NP Prep YOU
        NP -> ADJ N
        N ->""" + '| '.join(N) + """
        V ->""" + '| '.join(VERBS) + """
        ADJ ->""" + '| '.join(ADJ) + """
        Prep ->""" + '| '.join(PREP) + """
        YOU ->""" + '| '.join(YOU) + """
        """)
		print(self.grammar)
		self.sentences = [' '.join(s) for s in generate(self.grammar, n=1000)]

		print(self.sentences)
		print('%d different sentences possible.' % len(self.sentences))

gram = HighLevelGrammar()