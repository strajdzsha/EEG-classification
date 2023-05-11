import numpy as np
import os
import pickle
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from scipy.stats import kurtosis
from scipy.signal import butter, filtfilt
from typing import List
from config import *
from data_loader import DataLoader

class FeatureExtractor:
    """
    All features should be imlemented as a 
    method of this class
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
    def band_power(self, data: np.ndarray, band: str = None, freq: List[int] = None, order: int = 5, **kwargs):
        """
        Returns the power of signal in given band
        """
        assert band is not None or freq is not None, "Either band or freq must be specified"

        data = self.__filter(data, band, freq, order)
        return np.sum(data**2, axis=1) / data.shape[1]
    
    def __filter(self, data: np.ndarray, band: str = None, freq: List[int] = None, order: int = 5, **kwargs):
        """
        Filters the data with a band pass filter
        """
        if freq is None:
            nyq = 0.5 * FS # Nyquist frequency
            low, high = [x / nyq for x in FREQ_BANDS[band]]
        else:
            low, high = freq
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

    selector2 = BaselineSelector() # second selector
    selector2.selectFeatures(['band_power'], band='alpha')

    out2 = selector2.transform(arr)

    print(out1)
    print(out2)
    
    