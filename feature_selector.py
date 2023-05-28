import numpy as np
import os
import pickle
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from scipy.stats import kurtosis, skew
from scipy.signal import butter, filtfilt
from typing import List
from config import *
from data_loader import DataLoader

class FeatureExtractor:
    """
    All features should be imlemented as a 
    method of this class

    All methods should have the following signature:
    data (np.ndarray) - (n_channels, n_samples)
    **kwargs - additional parameters

    All methods should return a np.ndarray of shape (k*n_channels, ), 
    where k is the number of different features;
    take a look at the band_power method for an example
    """
    
    @staticmethod
    def mean(data: np.ndarray, **kwargs):
        """
        Returns the mean of the data
        """
        return np.mean(data, axis=1)
    
    @staticmethod
    def std(data: np.ndarray, **kwargs):
        """
        Returns the standard deviation of the data
        """
        return np.std(data, axis=1)
    
    @staticmethod
    def kurtosis(data: np.ndarray, **kwargs):
        """
        Returns the kurtosis of the data
        """
        return kurtosis(data, axis=1)
    
    @staticmethod
    def pca(data: np.ndarray, pca_components: int = None, **kwargs):
        """
        Returns the first n principal components of the data
        data (np.ndarray) - (n_channels, n_samples)
        """
        if pca_components is None:
            return data
        pca = PCA(n_components=pca_components)
        data = data.T  #(n_samples, n_channels)
        data = pca.fit_transform(data) # (n_samples, n_components)
        return data.T #(n_components, n_samples)
    
    @staticmethod
    def skewness(data: np.ndarray, **kwargs):
        """
        Returns the skewness of the data
        """
        return skew(data, axis=1)
    
    def band_power_alpha(self, data: np.ndarray, order: int = 4, **kwargs):
        """
        Returns the power of signal in alpha band
        """
        filtered_data = self.__filter(data, 'alpha', order)
        return np.sum(filtered_data**2, axis=1) / filtered_data.shape[1]
    
    def band_power_beta(self, data: np.ndarray, order: int = 4, **kwargs):
        """
        Returns the power of signal in beta band
        """
        filtered_data = self.__filter(data, 'beta', order)
        return np.sum(filtered_data**2, axis=1) / filtered_data.shape[1]
    
    def band_power_theta(self, data: np.ndarray, order: int = 4, **kwargs):
        """
        Returns the power of signal in theta band
        """
        filtered_data = self.__filter(data, 'theta', order)
        return np.sum(filtered_data**2, axis=1) / filtered_data.shape[1]
    
    def band_power_delta(self, data: np.ndarray, order: int = 4, **kwargs):
        """
        Returns the power of signal in delta band
        """
        filtered_data = self.__filter(data, 'delta', order)
        return np.sum(filtered_data**2, axis=1) / filtered_data.shape[1]
    
    def band_power_gamma(self, data: np.ndarray, order: int = 4, **kwargs):
        """
        Returns the power of signal in gamma band
        """
        filtered_data = self.__filter(data, 'gamma', order)
        return np.sum(filtered_data**2, axis=1) / filtered_data.shape[1]
    
    def hjorth_activity(self, data: np.ndarray, **kwargs):
        """
        Returns the activity of the data
        """
        return np.var(data, axis=1)
    
    def hjorth_mobility(self, data: np.ndarray, **kwargs):
        """
        Returns the mobility of the data
        """
        diff1 = np.diff(data, axis=1)
        return np.sqrt(np.var(diff1, axis=1) / np.var(data, axis=1))
    
    def hjorth_complexity(self, data: np.ndarray, **kwargs):
        """
        Returns the complexity of the data
        """
        diff1 = np.diff(data, axis=1)
        diff2 = np.diff(diff1, axis=1)
        return np.sqrt(np.var(diff2, axis=1) / np.var(diff1, axis=1)) / self.hjorth_mobility(data)
    
    def __filter(self, data: np.ndarray, band: str = None, order: int = 5, **kwargs):
        """
        Filters the data with a band pass filter
        """
        nyq = 0.5 * FS # Nyquist frequency
        low, high = [x / nyq for x in FREQ_BANDS[band]]
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

class FeatureSelector(ABC):
    """
    This class is used to choose the features to extract
    """
    def __init__(self) -> None:
        self.extractor = FeatureExtractor()
    
    def selectFeatures(self, features: List[str], **kwargs):
        """
        This method is used to select the features to extract
        """
        for f in features:
            assert hasattr(self.extractor, f), f"Feature {f} not supported"

        self.features = features
        self.kwargs = kwargs

    @abstractmethod
    def transform(self, data: np.ndarray, **kwargs):
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
    
    def transform(self, data: np.ndarray, **kwargs):
        
        data = self.extractor.pca(data, **self.kwargs)

        output = None
        for f in self.features:
            curr_feature = getattr(self.extractor, f)(data, **self.kwargs)
            if output is None:
                output = curr_feature
            else:
                output = np.concatenate((output, curr_feature))

        return output
    
class AnalysisSelector(FeatureSelector):
    """
    This class is used to extract the features used in analysis.
    Main difference is that this class returns the features in a dictionary.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, data: np.ndarray, **kwargs):

        data = self.extractor.pca(data, **self.kwargs)
        n_channels = data.shape[0]

        output = {}
        for f in self.features:
            curr_feature = getattr(self.extractor, f)(data, **self.kwargs)
            output[f] = curr_feature
        return output

    
if __name__ == "__main__":
    """
    Example usage of feature extractor
    """
    par_loader = DataLoader('./data/dataset', participants_ids=[0], seed=42)
    arr = par_loader[0]['data']
    print(arr.shape)

    selector1 = BaselineSelector() # first selector
    selector1.selectFeatures(['mean', 'kurtosis'], pca_components = 8)

    out1 = selector1.transform(arr)

    selector2 = AnalysisSelector() # second selector
    selector2.selectFeatures(['hjorth_mobility', 'band_power_gamma', 'band_power_delta', 'hjorth_complexity'], pca_components = 8)
    # selector2.selectFeatures(['band_power'], bands=['alpha', 'beta'])
    out2 = selector2.transform(arr)

    for key in out2:
        print(key, out2[key])
    
    