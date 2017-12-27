# Michael Wilson
from hmm import print_graph, Node

class Tagger(object):

	def __init__(self, model):
		self.model = model

	def __call__(self, sentence):
		self.reset(sentence)
		self.build_graph()
		self.score_graph()
		return list(zip(sentence, self.unwind()))

	def reset(self, words):
		self.words = words
		self.nodes = []

	def new_node(self, i, word, pos, prev_nodes):
		self.nodes.append(Node(len(self.nodes), i, word, pos, prev_nodes))

	def build_graph(self):
		self.nodes = []
		self.new_node(-1, None, None, [])
		prev_nodes = [self.nodes[0]]
		for i, word in enumerate(self.words):
			next_nodes = []
			for pos in self.model.parts(word):
				self.new_node(i, word, pos, prev_nodes)
				next_nodes.append(self.nodes[-1])
			prev_nodes = next_nodes
		self.new_node(self.nodes[-1].i + 1, None, None, prev_nodes)

	def edge_score(self, prev, next):
		if next.i != prev.i + 1:
			print("ERROR next.i != prev.i + 1")
			return
		return prev.score + self.model.tcost(prev.pos, next.pos) + self.model.ecost(prev.pos, prev.word)

	def score_node(self, node):
		node.best_prev = min(node.prev_nodes, key=lambda prev: self.edge_score(prev, node))
		node.score = self.edge_score(node.best_prev, node)

	def score_graph(self):
		self.nodes[0].score = 0
		for node in self.nodes[1:]:
			self.score_node(node)

	def unwind(self):
		best_prev = self.nodes[-1]
		tag_list = []
		while best_prev != None:
			tag_list.insert(0, best_prev.pos)
			best_prev = best_prev.best_prev
		return tag_list[1:-1]

# Testing

example_sents = [
    [('dogs', 'NNS'), ('bark', 'VB'), ('often', 'RB')],
    [('cats', 'NNS'), ('bark', 'VB'), ('dogs', 'NNS')],
    [('cats', 'NNS'), ('dogs', 'VB'), ('dogs', 'NNS'), ('often', 'RB')],
    [('bark', 'NNS'), ('often', 'RB'), ('dogs', 'VB'), ('cats', 'NNS')],
    [('bark', 'VB')],
    [('dogs', 'NNS'), ('bark', 'NNS')],
    [('dogs', 'NNS'), ('cats', 'NNS'), ('bark', 'VB')],
    [('often', 'RB'), ('cats', 'NNS'), ('bark', 'VB')]]

example_model = Model(example_sents)

example_model.display()

tagger = Tagger(example_model)
print(tagger.model.tprob(None, 'NNS'))

tagger.reset(['dogs', 'bark', 'often'])
print(tagger.words)

tagger.build_graph()
print_graph(tagger.nodes)

(n1, n2, n3) = tagger.nodes[1:4]
n1.score = .1
n2.score = .9
print(tagger.edge_score(n1, n3))
print(tagger.edge_score(n2, n3))

tagger.score_node(n3)
print(n3.score)
print(n3.best_prev)

tagger.score_graph()
print_graph(tagger.nodes)

print(tagger.unwind())

print(tagger.__call__(['dogs', 'bark', 'often']))
print(tagger(['dogs', 'bark', 'often']))