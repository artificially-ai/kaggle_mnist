import sys
import json

from model.mnist_cnn_classifier import MNISTCNNClassifier
from utils.s3 import S3Utils

if __name__ == '__main__':

    classifiers = {'cnn': MNISTCNNClassifier('model_output/cnn')}

    if len(sys.argv) < 2:
        print('Please, pass the model type you want to execute. for example, "cnn"')
        sys.exit(1)

    model_type = sys.argv[1]
    params = 'hyperparams_%s.json' % model_type
    print('Parameters file:', params)

    hyper_parameters = json.load(open('/data/%s' % params))
    mnist = classifiers[model_type]
    mnist.init(hyper_parameters)
    mnist.train_model()

    S3Utils.upload(model_type)
