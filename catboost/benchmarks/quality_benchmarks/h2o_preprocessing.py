from experiment import Experiment
import os
import argparse
import sys

class DataPreprocessor(Experiment):
    def convert_to_dataset(self, data, label, cat_cols=None):
        return (data, label)

def preprpocess(learning_task, train_path, test_path, cd_path, output_folder):
    experiment = DataPreprocessor(learning_task, train_path=train_path,
                                  test_path=test_path, cd_path=cd_path)
    X_train, y_train, X_test, y_test, cat_cols = experiment.read_data()
    cv_pairs, (dtrain, dtest) = experiment.split_and_preprocess(X_train.copy(), y_train,
                                                                X_test.copy(), y_test,
                                                                cat_cols, n_splits=2)

    for pool, filename in zip((dtrain, dtest), ('parsed_train', 'parsed_test')):
        with open(os.path.join(output_folder, filename), mode='wb') as f:
            for i in range(len(pool[0])):
                row_list = [pool[1][i]] + pool[0][i].tolist()
                row = "\t".join([str(j).encode('utf-8') for j in row_list])+"\n"
                f.write(row)

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('learning_task', choices=['classification', 'regression'])
    parser.add_argument('--train')
    parser.add_argument('--test')
    parser.add_argument('--cd')
    parser.add_argument('-o')
    return parser

if __name__ == "__main__":
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    preprpocess(namespace.learning_task, namespace.train, namespace.test, namespace.cd,
                            namespace.o)
