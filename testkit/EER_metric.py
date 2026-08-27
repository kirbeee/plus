import abc

class BaseMetric(object):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def compute(self, embeddings, labels, **kwargs):
        pass

class EERMetric(BaseMetric):
    def __init__(self, name):
        super(EERMetric, self).__init__("EER & Best ACC")

    def compute(self, embeddings, labels, **kwargs):
        print(embeddings.shape)
        print(labels.shape)
        return