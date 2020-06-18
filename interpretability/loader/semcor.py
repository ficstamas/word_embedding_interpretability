import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from interpretability.loader.semcat import SemCat
from interpretability.core.config import Config


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

    def get_tokens(self, in_file):
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
                yield synset_labels, lexname_labels, token_id, lemma, token.text.replace('-', '_')


def read(path: str, config: Config) -> (SemCat, list):
    """
    Loading SemCor
    :param path:
    :return:
    """
    vocab = {}
    c2i, i2c = {}, {}
    word_vector_indexes = {}
    eval_data = {}
    id = 0
    # Loading semcor
    for data in SemcorReader().get_tokens(path):
        if data[1].__len__() != 0:
            for lexname in data[1]:
                if lexname in vocab:
                    vocab[lexname].add(data[2])
                else:
                    if lexname != 'adj.ppl':
                        eval_data[lexname] = set()
                        vocab[lexname] = set()
                        vocab[lexname].add(data[2])
        word_vector_indexes[id] = data[2]
        id += 1
    # Loading eval data
    for data in SemcorReader().get_tokens(path.replace("data.xml", "eval.data.xml")):
        if data[1].__len__() != 0:
            for lexname in data[1]:
                lexname: str
                if lexname != 'adj.ppl':
                    id_stripped = '.'.join(data[2].split('.')[1:])
                    eval_data[lexname].add(id_stripped)

    id = 0
    # category to index
    for lexname in vocab:
        c2i[lexname] = id
        id += 1
    # index to category
    i2c = {v: k for k, v in c2i.items()}

    return SemCat(vocab, c2i, i2c, {}), word_vector_indexes
