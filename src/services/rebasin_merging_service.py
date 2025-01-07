from src.core.git_re_basin import GitReBasin


class ReBasinMergingService:
    def __init__(self, model1, model2):
        self.model1 = model1
        self.model2 = model2

    def merge(self):
        aligned_model2 = GitReBasin.align_permutation(self.model1, self.model2)
        return GitReBasin.merge_models(self.model1, self.model2, aligned_model2)
