from tqdm import tqdm

class DummyTqdmFile:
    """Dummy file-like that will write to tqdm
    https://github.com/tqdm/tqdm/issues/313
    """
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file, end='')

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

