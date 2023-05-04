import numpy as np
import os
import pickle
from abc import abstractmethod
from sklearn.decomposition import PCA
from scipy.stats import kurtosis
from typing import List

class FeatureExtractor:
    """
    All features should be imlemented as a 
    method of this class
    """

    @staticmethod
    def mean(data: np.ndarray):
        """
        Returns the mean of the data
        """
        return np.mean(data, axis=1)
    
    @staticmethod
    def std(data: np.ndarray):
        """
        Returns the standard deviation of the data
        """
        return np.std(data, axis=1)
    
    @staticmethod
    def kurtosis(data: np.ndarray):
        """
        Returns the kurtosis of the data
        """
        return kurtosis(data, axis=1)
    
    @staticmethod
    def pca(data: np.ndarray, n_components: int):
        """
        Returns the first n principal components of the data
        data (np.ndarray) - (n_channels, n_samples)
        """
        pca = PCA(n_components=n_components)
        data = data.T  #(n_samples, n_channels)
        data = pca.fit_transform(data) # (n_samples, n_components)
        return data.T #(n_components, n_samples)

class FeatureSelector:
    """
    This class is used to choose the features to extract
    """
    def __init__(self) -> None:
        self.extractor = FeatureExtractor()
    
    def selectFeatures(self, features: List[str], pca_components: int = None):
        """
        This method is used to select the features to extract
        """
        for f in features:
            assert hasattr(self.extractor, f), f"Feature {f} not supported"
        
        self.features = features
        self.pca_components = pca_components

    @abstractmethod
    def transform(self, data: np.ndarray):
        """
        This method is used to transform the data into features
        """
        pass


class BaselineSelector(FeatureSelector):
    """
    This class is used to extract the baseline features
    """
    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, data: np.ndarray):
        if self.pca_components:
            data = self.extractor.pca(data, self.pca_components)

        output = None
        for f in self.features:
            curr_feature = getattr(self.extractor, f)(data)
            if output is None:
                output = curr_feature
            else:
                output = np.concatenate((output, curr_feature))

        return output
    
if __name__ == "__main__":
    """
    Example usage of feature extractor
    """
    arr = np.ones((4, 5))

    arr[:, 0] = 2
    arr[1, 1] = 3
    arr[2, 2] = 4

    selector1 = BaselineSelector() # first selector
    selector1.selectFeatures(['mean', 'kurtosis'], pca_components=2)

    out1 = selector1.transform(arr)

    selector2 = BaselineSelector() # second selector
    selector2.selectFeatures(['mean', 'std'])

    out2 = selector2.transform(arr)

    print(out1)
    print(out2)
    
    
    