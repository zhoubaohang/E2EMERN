import codecs
import numpy as np
from keras_bert import Tokenizer


class EntityBase(object):
    def __init__(self, bert_path: str):
        self.base_path = "./entitybase"
        self.mesh_path = f"{self.base_path}/mesh.tsv"
        self.omim_path = f"{self.base_path}/omim.tsv"
        self.__bert_path = f"{bert_path}/vocab.txt"
        self.__tokenizer = self.__load_vocabulary()

        self.entities = {}

        self.__load_mesh()
        self.__load_omim()

    def __load_vocabulary(self):
        token_dict = {}

        with codecs.open(self.__bert_path, "r", "utf8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)

        return Tokenizer(token_dict)

    def __tokenize_entity(self, entity):
        ind, _ = self.__tokenizer.encode(first=entity)
        return ind

    def random_entity(self, idxs):
        """
        select one entity which are not in the list
        generate negative sample for entity normalization
        """
        dicts = list(self.entities.keys())
        idx = np.random.choice(dicts)
        while idx in idxs:
            idx = np.random.choice(dicts)
        return self.getItem(idx)

    def generate_candidates(self, sentence, idxs):
        cands = []
        parsed_sentence = self.__tokenize_entity(" ".join(sentence))
        for k, v in self.entities.items():
            ent = v[1]
            if len(set(ent[1:-1]) & set(parsed_sentence[1:-1])) > 2 and k not in idxs:
                cands.append(k)
        return cands

    def getVocabs(self):
        names = []
        for k, v in self.entities.items():
            names += v[0]
        return names

    def getItem(self, uid):
        result = self.entities.get(uid, ["none"])
        # if result == 'none':
        #     print(f"UNDEFINED WARNING! [{uid}]")
        return result[0]

    @property
    def Entity(self):
        return self.entities

    def __load_mesh(self):
        data = np.loadtxt(self.mesh_path, delimiter="\t", dtype=str)

        for ele in data.tolist():
            self.entities[ele[0]] = (ele[1], self.__tokenize_entity(ele[1]))

    def __load_omim(self):
        data = np.loadtxt(self.omim_path, delimiter="\t", dtype=str, encoding="utf-8")

        for ele in data.tolist():
            self.entities[f"OMIM:{ele[0]}"] = (ele[1], self.__tokenize_entity(ele[1]))


if __name__ == "__main__":
    eb = EntityBase()
    vocabs = eb.getVocabs()