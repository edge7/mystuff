class FeatureSelection(object):
    def __init__(self, df, path):
        self.file_descriptor = open(path + "_featureSelection", "a")
        self.cols = df.columns.tolist()
        self.cols.remove("target")
        self.feature_importance = {}

    def write_feature_importance(self, model):
        feature_import = model.feature_importances_.tolist()
        new_list = [(importance, name) for name, importance in zip(self.cols, feature_import)]
        sorted_by_importance = sorted(new_list, key=lambda tup: tup[0], reverse=True)
        for importance, name in sorted_by_importance:
            res = self.feature_importance.get(name, [])
            res.append(float(importance))
            self.feature_importance[name] = res

    def consolidate_feature_importance(self):
        l = []
        for name, lis in self.feature_importance.items():
            tot = float(sum(lis))
            count = float(len(lis))
            mean = tot / count
            l.append((name, mean))

        tw = sorted(l, key=lambda tup: tup[1], reverse=True)
        to_write = ''
        for t in tw:
            to_write += str(t[0]) + ':   ' + str(t[1])
            to_write += '\n'

        self.file_descriptor.write('\n\n\n\n\n\n\n\n Writing Feature IMPORTANCE:\n\n')
        self.file_descriptor.write(to_write)
        self.feature_importance = {}
        self.file_descriptor.flush()