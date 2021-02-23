import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from interpretability.reader.semcat import SemCat
from interpretability.core.config import Config


class Semcor(SemCat):
    def __init__(self, vocab, c2i, i2c, eval_data, word_vector_tokens):
        super(Semcor, self).__init__(vocab, c2i, i2c, eval_data)
        self.word_vector_tokens = word_vector_tokens


class SemcorReader:
    @staticmethod
    def get_labels(key_file):
        id_to_gold, sense_to_id = {}, {}
        with open(key_file) as f:
            for l in f:
                position_id, *senses = l.split()
                id_to_gold[position_id] = senses

                for s in senses:
                    if s not in sense_to_id:
                        sense_to_id[s] = [len(sense_to_id), 1]
                    else:
                        sense_to_id[s][1] += 1
        return id_to_gold, sense_to_id

    def get_tokens(self, in_file, evaldata=False):
        etalons, _ = self.get_labels(in_file.replace('data.xml', 'gold.key.txt'))
        root = ET.parse(in_file).getroot()
        for s in root.findall('text/sentence'):
            for token in list(s):
                pos_tag = token.attrib['pos']
                pos_tag_wn = 'r'
                if pos_tag != "ADV": pos_tag_wn = pos_tag[0].lower()
                token_id = None
                synset_labels, lexname_labels = [], []
                if 'id' in token.attrib:
                    token_id = token.attrib['id']
                    for sensekey in etalons[token.attrib['id']]:
                        synset = wn.lemma_from_key(sensekey).synset()
                        synset_labels.append(synset.name())
                        lexname_labels.append(synset.lexname())
                lemma = '{}.{}'.format(token.attrib['lemma'], pos_tag_wn)
                yield synset_labels, lexname_labels, token_id, lemma, token.text.replace('-', '_'), s.attrib["id"].split('.')[0] if evaldata else None


def read(path: str, config: Config) -> (Semcor, list, list, dict):
    """
    Loading SemCor
    :param path:
    :return:
    """
    vocab = {}
    c2i, i2c = {}, {}
    word_vector_indexes = {}
    word_vector_tokens = {}
    eval_data = {}
    eval_vector_indexes = {}
    id = 0
    # Loading semcor
    for data in SemcorReader().get_tokens(path):
        if data[1].__len__() != 0:
            for lexname in set(data[1]):
                if lexname in vocab:
                    vocab[lexname].add(data[4])
                else:
                    eval_data[lexname] = set()
                    vocab[lexname] = set()
                    vocab[lexname].add(data[4])
        word_vector_indexes[id] = list(set(data[1]))[0] if data[1].__len__() != 0 else '<unknown>'
        word_vector_tokens[id] = data[1]
        id += 1

    # separates the concatenated semeval datasets
    border = {}

    id = 0
    # Loading eval data
    for data in SemcorReader().get_tokens(config.semantic_categories.test_words_path, evaldata=True):
        if data[1].__len__() != 0:
            for lexname in set(data[1]):
                lexname: str
                eval_data[lexname].add(data[4])
        eval_vector_indexes[id] = list(set(data[1]))[0] if data[1].__len__() != 0 else '<unknown>'
        border[data[-1]] = id
        id += 1

    id = 0
    # category to index
    for lexname in vocab:
        c2i[lexname] = id
        id += 1
    # index to category
    i2c = {v: k for k, v in c2i.items()}
    config.logger.info(f"Semcor is loaded!")
    return Semcor(vocab, c2i, i2c, eval_data, word_vector_tokens), word_vector_indexes, eval_vector_indexes, border


def read_with_path(train_path: str, test_path: str) -> (Semcor, list, list):
    """
    Loading SemCor from given path (no config object required, to work with outer scripts)
    :param path:
    :param train_path:
    :param test_path:
    :return:
    """
    vocab = {}
    c2i, i2c = {}, {}
    word_vector_indexes = {}
    word_vector_tokens = {}
    eval_data = {}
    eval_vector_indexes = {}
    id = 0
    # Loading semcor
    for data in SemcorReader().get_tokens(train_path):
        if data[1].__len__() != 0:
            for lexname in set(data[1]):
                if lexname in vocab:
                    vocab[lexname].add(data[4])
                else:
                    # if lexname != 'adj.ppl':
                    eval_data[lexname] = set()
                    vocab[lexname] = set()
                    vocab[lexname].add(data[4])
        word_vector_indexes[id] = list(set(data[1]))[0] if data[1].__len__() != 0 else '<unknown>'
        # if word_vector_indexes[id] == 'adj.ppl':
        #     word_vector_indexes[id] = '<unknown>'
        word_vector_tokens[id] = data[1]
        id += 1

    id = 0
    # Loading eval data
    for data in SemcorReader().get_tokens(test_path):
        if data[1].__len__() != 0:
            for lexname in set(data[1]):
                lexname: str
                # if lexname != 'adj.ppl':
                eval_data[lexname].add(data[4])
        eval_vector_indexes[id] = list(set(data[1]))[0] if data[1].__len__() != 0 else '<unknown>'
        # if eval_vector_indexes[id] == 'adj.ppl':
        #     eval_vector_indexes[id] = '<unknown>'
        id += 1

    id = 0
    # category to index
    for lexname in vocab:
        c2i[lexname] = id
        id += 1
    # index to category
    i2c = {v: k for k, v in c2i.items()}
    return Semcor(vocab, c2i, i2c, eval_data, word_vector_tokens), word_vector_indexes, eval_vector_indexes
