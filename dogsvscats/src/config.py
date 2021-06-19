sgd_config = {
         'BATCH_SIZE': 32,
         'EVAl_AFTER_EPOCHS':10,
         'dataset' : 'dogsvscats',
         'learning_rate' : 0.001,
         'momentum' : 0.9,
         'optimizer' : 'sgd',
         'architecture' : 'resnet34'
         }

adam_config = {
         'BATCH_SIZE': 32,
         'EVAl_AFTER_EPOCHS':10,
         'dataset' : 'dogsvscats',
         'learning_rate' : 0.001,
         'optimizer' : 'Adam',
         'architecture' : 'resnet18',
         'train_file' : "/home/lovekesh/Developer/nnet_experiments/dogsvscats/data/train.txt",
         'valid_file' : "/home/lovekesh/Developer/nnet_experiments/dogsvscats/data/valid.txt"
         }