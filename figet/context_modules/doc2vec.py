# -*- coding: utf-8 -*-

from __future__ import print_function

import random
import gensim

from figet import utils



class Doc2Vec(object):

    def __init__(self, train_data=None, test_data=None, save_path="doc2vec.pt", log=utils.get_logging()):
        self.model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
        self.train_data = train_data
        self.test_data = test_data
        self.save_path = save_path
        self.log = log

    def train(self):
        self.log.info("Loading the training data from %s." %self.train_data)
        train_corpus = self.read_corpus(self.train_data)
        self.log.info("Building vocab.")
        self.model.build_vocab(train_corpus)
        self.log.info("Start training.")
        self.model.train(train_corpus, total_examples=self.model.corpus_count, epochs=self.model.iter)
        self.log.info("Saving the model to %s." %self.save_path)
        self.model.save(self.save_path)

    def test(self):
        train_corpus = list(self.read_corpus(self.train_data))
        test_corpus = list(self.read_test_corpus(self.test_data))
        doc_id = random.randint(0, len(test_corpus))
        inferred_vector = self.model.infer_vector(test_corpus[doc_id])
        sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))

        # Compare and print the most/median/least similar documents from the train corpus
        self.log.info('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
        self.log.info(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % self.model)
        for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
            self.log.info(u'%s %s: «%s»\n' % (
                label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

    def transform(self, doc):
        return self.model.infer_vector(gensim.utils.simple_preprocess(doc))

    def load(self):
        self.log.info("Loading the pretrained model from %s." %self.save_path)
        self.model = gensim.models.doc2vec.Doc2Vec.load(self.save_path)

    def read_corpus(self, fname):
        with open(fname) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line == "":
                    continue
                line = line.replace("\\n", " ")
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

    def read_test_corpus(self, fname):
        with open(fname) as f:
            for i, line in enumerate(f.read().strip().split('\n\n')):
                line = line.strip()
                if line == "":
                    continue
                line = line.replace("\\n", " ")
                yield gensim.utils.simple_preprocess(line)


def main(args):
    m = Doc2Vec(args.corpus, args.test, args.model)
    m.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("doc2vec")
    parser.add_argument("-corpus", help="The corpus path.")
    parser.add_argument("-model", default="doc2vec.pt", help="The model path.")
    parser.add_argument("-test", help="The test corpus path.")

    args = parser.parse_args()
    main(args)
