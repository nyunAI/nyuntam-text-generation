from text_generation.core.job import LMJob


# nyuntam
from nyuntam.algorithm import Algorithm


class AQLM(Algorithm):

    def __init__(self, job: LMJob, **kwargs):
        self.job = job
