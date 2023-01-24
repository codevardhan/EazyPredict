# EazyPredict

"Welcome to the 'EazyPredict' module, where we make predictions as simple as 1, 2, 3... and the jokes are always a 4." - ChatGPT when asked for a joke to begin this module documentation. :P

EazyPredict serves as a quick way to try out multiple prediction algorithms on data while writing as few lines as possible. It also provides the possibility to create an ensemble of the top models (Not yet implemented)

The 'EazyPredict' module was heavily inspired by [LazyPredict](https://github.com/shankarpandala/lazypredict). This module varies slightly in terms of its functionality and intended use, as outlined in the following ways:

- The 'EazyPredict' module utilizes a limited number of prediction algorithms (around 9) in order to minimize memory usage and prevent potential issues on platforms such as Kaggle.

- The models can be saved to an output folder at the user's discretion and are returned as a dictionary, allowing for easy addition of custom hyperparameters.

- The top 5 models are selected to create an ensemble using a voting classifier (this feature is not yet implemented).
