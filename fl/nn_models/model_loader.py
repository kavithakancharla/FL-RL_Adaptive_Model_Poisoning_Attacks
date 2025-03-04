def load_model(dataset_name):
    if dataset_name == 'mnist':
        from fl.nn_models.mnist_nn import MNSIT_NN
        return MNSIT_NN()
    else:
        raise ValueError(f'No model found for dataset: {dataset_name}')