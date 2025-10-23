class InferenceChain:

    def __init__(self, data):
        self._data = data

    def calibrate(self, rundir: str=None):
        raise RuntimeError("Method must be subclassed as InferenceChain is a base class")

    def predict(self):
        raise RuntimeError("Method must be subclassed as InferenceChain is a base class")
