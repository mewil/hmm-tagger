# Michael Wilson
from math import log10

class Model (object):

    def __init__ (self, tsents):
        self.train(tsents)

    def train (self, tsents):
        self.transitions = {}
        self.emissions = {}
        self.tagcounts = {} 
        self.count_emission(None, None)
        self.collect_counts(tsents)
        self.normalize()
        self.tagcounts = None

    def collect_counts (self, tsents):
        for sent in tsents:
            prev_tag = None
            for (w,t) in sent:
                self.count_emission(t, w)
                self.count_transition(prev_tag, t)
                prev_tag = t
            self.count_transition(prev_tag, None)

    def count_transition (self, prev, next):
        if prev in self.transitions:
            row = self.transitions[prev]
        else:
            self.transitions[prev] = row = {}
        if next in row:
            row[next] += 1
        else:
            row[next] = 1

    def count_emission (self, t, w):
        if w in self.emissions:
            col = self.emissions[w]
        else:
            self.emissions[w] = col = {}
        if t in col:
            col[t] += 1
        else:
            col[t] = 1

        if t in self.tagcounts:
            self.tagcounts[t] += 1
        else:
            self.tagcounts[t] = 1

    def normalize (self):
        for row in self.transitions.values():
            N = sum(row.values())
            for prev in row:
                row[prev] /= N

        for col in self.emissions.values():
            for t in col:
                N = self.tagcounts[t]
                col[t] /= N


    # Scoring

    def tprob (self, prev, next):
        if prev in self.transitions:
            row = self.transitions[prev]
            if next in row:
                return self.transitions[prev][next]
        return 0.0

    def eprob (self, pos, word):
        if word in self.emissions:
            col = self.emissions[word]
            if pos in col:
                return col[pos]
        return 0.0

    def tcost (self, prev, next):
        p = self.tprob(prev, next)
        if p == 0.0: return float('inf')
        else: return -log10(p)

    def ecost (self, pos, word):
        p = self.eprob(pos, word)
        if p == 0.0: return float('inf')
        else: return -log10(p)

    def parts (self, word):
        return sorted(self.emissions[word].keys())


    # Debugging

    def display (self):
        print('{:13} {:>7} {:>7}'.format('Transitions', 'prob', 'cost'))
        for pw in self.transitions:
            print('    {}:'.format(pw))
            row = self.transitions[pw]
            for nw in row:
                print('{:8}{:<5} {:7.4f} {:7.4f}'.format('', str(nw), row[nw], -log10(row[nw])))
        print('Emissions')
        for w in self.emissions:
            print('{:4}{}:'.format('', w))
            col = self.emissions[w]
            for t in sorted(col):
                print('{:8}{:<5} {:7.4f} {:7.4f}'.format('', str(t), col[t], -log10(col[t])))


def evaluate (tagger, test):
    ntoks = 0
    nerrs = 0
    for tsent in test:
        sent = [w for (w,t) in tsent]
        psent = tagger(sent)
        for i in range(len(tsent)):
            ntoks += 1
            if i < len(psent) or psent[i] != tsent[i]:
                nerrs += 1
    return 1 - nerrs/ntoks


class Node (object):

    def __init__ (self, index, i, word, pos, prev_nodes):

        self.index = index
        self.i = i
        self.word = word
        self.pos = pos
        self.prev_nodes = prev_nodes

        self.score = None
        self.best_prev = None

    def __repr__ (self):
        return '<Node {}>'.format(self.index)


def print_graph (nodes):
    print('Graph:')
    print('     {:3}  {:>2} {:10} {:5} {:8} {:2} {:>7}'.format(
            'ind', 'i', 'word', 'pos', 'prevs', 'bp', 'score'))
    i = -float('inf')
    for node in nodes:
        if node.i > i:
            i = node.i
            print('    ---------------------------------------------')
        print_node(node)

def print_node (node):
    if node.score is None: v = ''
    else: v = format(node.score, '7.4f')
    if node.best_prev: prev = node.best_prev.index
    else: prev = ''
    print('    [{:3}] {:2} {:<10} {:<5} {:8} {:2} {:7}'.format(
            node.index,
            node.i,
            (node.word or '')[:10],
            (node.pos or '')[:5],
            ','.join(str(p.index) for p in node.prev_nodes)[:8],
            prev,
            v))
