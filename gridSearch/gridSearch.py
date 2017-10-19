

class GridSearchCustomModel(object):
    model = None
    params = None

    def __init__(self, model, params):
        self.model = model
        self.params = params


