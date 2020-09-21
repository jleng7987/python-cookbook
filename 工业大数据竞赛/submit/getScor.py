import pandas as pd

class score:
    def __init__(self, path):
        self.path = path

    def path_data(self):
        df = pd.read_csv(self.path)

        return df
