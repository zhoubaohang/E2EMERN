import numpy as np

class EntityBase(object):

    def __init__(self):
        self.base_path = './entitybase'
        self.mesh_path = f"{self.base_path}/mesh.tsv"
        self.omim_path = f"{self.base_path}/omim.tsv"

        self.entities = {}

        self.__load_mesh()
        self.__load_omim()

    def random_entity(self, idxs):
        '''
            select one entity which are not in the list
            generate negative sample for entity normalization
        '''
        dicts = list(self.entities.keys())
        idx = np.random.choice(dicts)
        while idx in idxs:
            idx = np.random.choice(dicts)
        return self.getItem(idx)

    def getVocabs(self):
        names = []
        for k, v in self.entities.items():
            names += v
        return names

    def getItem(self, uid):
        result = self.entities.get(uid, 'none')
        # if result == 'none':
        #     print(f"UNDEFINED WARNING! [{uid}]") 
        return result
    
    @property
    def Entity(self):
        return self.entities

    def __load_mesh(self):
        data = np.loadtxt(self.mesh_path, delimiter='\t', dtype=str)

        for ele in data.tolist():
            self.entities[ele[0]] = ele[1]

    def __load_omim(self):
        data = np.loadtxt(self.omim_path, delimiter='\t', dtype=str, encoding='utf-8')

        for ele in data.tolist():
            self.entities[f"OMIM:{ele[0]}"] = ele[1]

if __name__ == '__main__':
    eb = EntityBase()
    vocabs = eb.getVocabs()