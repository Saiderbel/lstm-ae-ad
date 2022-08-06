import torch

class ModelWrapper(torch.nn.Module):

    def __init__(self, ad_module):
        super(ModelWrapper, self).__init__()
        self.ad_module = ad_module

    def forward(self, x):
        outputs = self.ad_module(x)
        diff = torch.mean(torch.abs(self.flatten(outputs) - self.flatten(x)), axis=1)

        return diff.view((-1, 1))

    # function to flatten the sampled sequence
    def flatten(self, X):
        '''
        Flatten a 3D array.

        Input
        X            A 3D array for lstm, where the array is sample x timesteps x features.

        Output
        flattened_X  A 2D array, sample x features.
        '''
        flattened_X = torch.empty((X.shape[0], X.shape[2]))  # sample x features array.
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1] - 1), :]
        return (flattened_X)