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
    
    def band_power(self, data: np.ndarray, bands: List[str] = None, order: int = 5, **kwargs):
        """
        Returns the power of signal in given band
        """
        output = None
        if bands is None:
            bands = FREQ_BANDS.keys()
        for band in bands:
            data = self.__filter(data, band, order)
            if output is None:
                output = np.sum(data**2, axis=1) / data.shape[1]
            else:
                output = np.concatenate((output, np.sum(data**2, axis=1) / data.shape[1]))
        return output.flatten()

    def hjorth_params(self, data: np.ndarray, params: List[str], **kwargs):
        """
        Returns the Hjorth parameters of the data
        """
        output = None
        for param in params:
            if param == 'activity':
                curr_feature = self.__activity(data)
            elif param == 'mobility':
                curr_feature = self.__mobility(data)
            elif param == 'complexity':
                curr_feature = self.__complexity(data)
            else:
                raise Exception(f"Parameter {param} not supported")
            if output is None:
                output = curr_feature
            else:
                output = np.concatenate((output, curr_feature))
        return output.flatten()

    def __activity(self, data: np.ndarray):
        """
        Returns the activity of the data
        """
        return np.var(data, axis=1)
    
    def __mobility(self, data: np.ndarray):
        """
        Returns the mobility of the data
        """
        diff1 = np.diff(data, axis=1)
        return np.sqrt(np.var(diff1, axis=1) / np.var(data, axis=1))
    
    def __complexity(self, data: np.ndarray):
        """
        Returns the complexity of the data
        """
        diff1 = np.diff(data, axis=1)
        diff2 = np.diff(diff1, axis=1)
        return np.sqrt(np.var(diff2, axis=1) / np.var(diff1, axis=1)) / self.__mobility(data)
    
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
    
    def transform(self, data: np.ndarray):

        data = self.extractor.pca(data, **self.kwargs)
        n_channels = data.shape[0]

        output = {}
        for f in self.features:
            if f == 'band_power':
                bands = []
                if 'bands' not in self.kwargs: bands = FREQ_BANDS.keys()
                else: bands = self.kwargs['bands']
                curr_feature = getattr(self.extractor, f)(data, **self.kwargs)
                for i, param in enumerate(bands):
                    output[f'{f}_{param}'] = curr_feature[i*n_channels:(i+1)*n_channels]
            elif f == 'hjorth_params':
                params = []
                if 'params' not in self.kwargs: params = ['activity', 'mobility', 'complexity']
                else: params = self.kwargs['params']
                curr_feature = getattr(self.extractor, f)(data, **self.kwargs)
                for i, param in enumerate(params):
                    output[f'{f}_{param}'] = curr_feature[i*n_channels:(i+1)*n_channels]
            else:
                curr_feature = getattr(self.extractor, f)(data, **self.kwargs)
                output[f] = curr_feature
        return output

    
if __name__ == "__main__":
    """
    Example usage of feature extractor
    """
    par_loader = DataLoader('./data/dataset', participants_ids=[0])
    arr = par_loader[0]['data']
    print(arr.shape)

    selector1 = BaselineSelector() # first selector
    selector1.selectFeatures(['mean', 'kurtosis'], pca_components = 2)

    out1 = selector1.transform(arr)

    selector2 = AnalysisSelector() # second selector
    selector2.selectFeatures(['hjorth_params', 'mean', 'band_power', 'kurtosis'], pca_components = 2, params=['activity', 'mobility'], bands=['alpha', 'beta'])
    # selector2.selectFeatures(['band_power'], bands=['alpha', 'beta'])
    out2 = selector2.transform(arr)

    for key in out2:
        print(key, out2[key])
    
    