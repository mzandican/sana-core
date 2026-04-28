class MetaController:
    def __init__(self):
        self.stress = 0.01
        self.lr = 0.001

    def adjust(self, loss):
        if loss > 1.0:
            self.stress *= 1.1
            self.lr *= 0.9
        else:
            self.stress *= 0.9
            self.lr *= 1.05

        self.stress = min(max(self.stress, 0.001), 0.1)
        self.lr = min(max(self.lr, 1e-5), 1e-2)

        return self.stress, self.lr
