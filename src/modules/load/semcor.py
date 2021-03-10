import xml.etree.ElementTree as ET
from nltk.corpus import wordnet as wn
from src.modules.utilities.labels import Labels
import os
from src.modules.utilities.logging import Logger
import tqdm


class SemcorReader:
    def __init__(self, path: str, tqdm_params=None):
        self.path = path
        self.log = Logger().logger
        self.log.debug(f"Initiating SemcorReader with path {path}")
        self._etalons, _ = self.get_labels(path.replace('data.xml', 'gold.key.txt'))
        self._root = ET.parse(path).getroot()
        self._sentences = self._root.findall('text/sentence')
        self._pbar = None

        if tqdm_params is None:
            self._tqdm_params = {
                'total': len(self),
                'unit': " sentence",
                'desc': "[SemcorReader]\t:"
            }
        else:
            self._tqdm_params = tqdm_params

    def __len__(self):
        return len(self._sentences)

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

    def __repr__(self):
        return f"<SemcorReader path={self.path}>"

    def get_tokens(self):
        self.log.info(f"Reading files with {self}")
        # to avoid breaking the logging and progressbar + this way I can reuse the same object
        if self._pbar is None:
            self._pbar = tqdm.tqdm(**self._tqdm_params)
        etalons = self._etalons
        for s in self._sentences:
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
                yield synset_labels, lexname_labels, token_id, lemma, token.text.replace('-', '_'), s.attrib["id"].split('.')[0]
            self._pbar.update()
        self._pbar.close()
        self._pbar = None


def lexname_as_label(train_labels_path: str, test_labels_path=None) -> [Labels, Labels]:
    log = Logger().logger

    if not os.path.exists(train_labels_path):
        raise FileNotFoundError(f"File not found at {train_labels_path}")
    log.info(f"Loading train labels: {train_labels_path}")

    labels = []
    dataset = []
    for token in SemcorReader(train_labels_path).get_tokens():
        labels.append(token[1])
        dataset.append(token[-1])

    train = Labels(labels, dataset)

    if test_labels_path is None:
        return train, None

    if not os.path.exists(test_labels_path):
        raise FileNotFoundError(f"File not found at {test_labels_path}")
    log.info(f"Loading test labels: {test_labels_path}")

    labels = []
    dataset = []
    for token in SemcorReader(test_labels_path).get_tokens():
        labels.append(token[1])
        dataset.append(token[-1])

    test = Labels(labels, dataset)
    return train, test


def lexname_as_label_eval(test_labels_path: str) -> Labels:
    log = Logger().logger

    if not os.path.exists(test_labels_path):
        raise FileNotFoundError(f"File not found at {test_labels_path}")
    log.info(f"Loading test labels: {test_labels_path}")

    labels = []
    dataset = []
    for token in SemcorReader(test_labels_path).get_tokens():
        labels.append(token[1])
        dataset.append(token[-1])

    test = Labels(labels, dataset)
    return test
