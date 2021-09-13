import sklearn

def confusion_matrix(preds, true_labels):
   ... 


def get_one_batch(datamodule):
    datamodule.setup()
    return next(iter(datamodule.train_dataloader()))
