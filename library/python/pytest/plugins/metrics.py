class TestMetrics:
    metrics = {}

    def __getitem__(self, key):
        return self.metrics.__getitem__(key)

    def __setitem__(self, key, value):
        return self.metrics.__setitem__(key, value)

    def get(self, key):
        return self.metrics.get(key)


test_metrics = TestMetrics()
