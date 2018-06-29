def pytest_collection_modifyitems(items, config):
    with open("test-list", "w") as f:
        for item in items:
            f.write(item.nodeid + "\n")
