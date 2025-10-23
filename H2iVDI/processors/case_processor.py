import logging

from H2iVDI.core.logger import create_logger

class CaseProcessor:

    def __init__(self):
        self._logger = logging.getLogger("H2iVDI")
        if not hasattr(self._logger, "debugL2"):
            self._logger = create_logger()
        self._data = None

    def prepro(self, **kwargs):
        raise RuntimeError("Method must be subclassed as CaseProcessor is a base class")

    def run(self, **kwargs):
        raise RuntimeError("Method must be subclassed as CaseProcessor is a base class")

    def postpro(self, **kwargs):
        raise RuntimeError("Method must be subclassed as CaseProcessor is a base class")
