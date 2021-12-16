import os
import dill, pickle
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def save_pickle(obj, filename, use_dill=False, protocol=4, create_folder=True):
    """ Basic pickle/dill dumping.

    Given a python object and a filename, the method will save the object under that filename.

    Args:
        obj (python object): The object to be saved.
        filename (str): Location to save the file.
        use_dill (bool): Set True to save using dill.
        protocol (int): Pickling protocol (see pickle docs).
        create_folder (bool): Set True to create the folder if it does not already exist.

    Returns:
        None
    """
    if create_folder:
        _create_folder_if_not_exist(filename)

    # Save
    with open(filename, 'wb') as file:
        if not use_dill:
            pickle.dump(obj, file, protocol=protocol)
        else:
            dill.dump(obj, file)


def load_pickle(filename):
    """ Basic dill/pickle load function.

    Args:
        filename (str): Location of the object.

    Returns:
        python object: The loaded object.
    """

    with open(filename, 'rb') as file:
        if filename.split('.')[-1] == 'dill':
            obj = dill.load(file)
        else:
            obj = pickle.load(file)
    return obj


def _create_folder_if_not_exist(filename):
    """ Makes a folder if the folder component of the filename does not already exist. """
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def compute_prediction_utility(labels, predictions, dt_early=-12, dt_optimal=-6, dt_late=3.0, max_u_tp=1, min_u_fn=-2,
                               u_fp=-0.05, u_tn=0, check_errors=True, return_all_scores=False):
    # Check inputs for errors.
    if check_errors:
        if len(predictions) != len(labels):
            raise Exception('Numbers of predictions and labels must be the same.')

        for label in labels:
            if not label in (0, 1):
                raise Exception('Labels must satisfy label == 0 or label == 1.')

        for prediction in predictions:
            if not prediction in (0, 1):
                raise Exception('Predictions must satisfy prediction == 0 or prediction == 1.')

        if dt_early >= dt_optimal:
            raise Exception('The earliest beneficial time for predictions must be before the optimal time.')

        if dt_optimal >= dt_late:
            raise Exception('The optimal time for predictions must be before the latest beneficial time.')

    # Does the patient eventually have sepsis?
    if np.any(labels):
        is_septic = True
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        is_septic = False
        t_sepsis = float('inf')

    n = len(labels)

    # Define slopes and intercept points for utility functions of the form
    # u = m * t + b.
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    # Compare predicted and true conditions.
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            # TP
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            # FP
            elif not is_septic and predictions[t]:
                u[t] = u_fp
            # FN
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            # TN
            elif not is_septic and not predictions[t]:
                u[t] = u_tn

    # Find total utility for patient.
    if return_all_scores:
        return u
    else:
        return np.sum(u)



class TimeSeriesDataset(Dataset):
    """A class for working with variable length time-series data in tensor format.

    This class has been built out of a desire to perform matrix operations time-series data where the time-dimension has
    variable length. This assumes we have N time-series each with C channels and length L_i, where L_i can vary between
    series. This class will create a tensor of shape [N, L_max, C] where L_max is the longest time-series in the data
    and provide methods for keeping track of the original time-series components, whilst allowing the data to be
    manipulated through matrix operations.

    This class should be used rather than simply a nan-filled tensor up to max length when you need to do the following:
        - Access columns by column name, rather than index
        - Convert back to the original lengths after computations

    # TODO: A lot of functionality can be added to this.
        - When getting dataset[column_name] it would be good to return a TimeSeriesDataset instance instead of a tensor
        we then need to implement how to add, subtract, divide etc with this class.
    """
    def __init__(self, data=None, columns=None):
        """
        Args:
            data (list): A list of variable length tensors. The shape must be [1, L_i, C] where L_i is the variable time
                         dimension, if list length is N this will create an [N, L_max, C] tensor.
            columns (list): List of column names of length C.
        """
        if data is not None:
            self.data = torch.nn.utils.rnn.pad_sequence(data, padding_value=np.nan, batch_first=True)
            self.lengths = [d.size(0) for d in data]
            self.columns = columns

            # Error handling
            self.init_assertions()

        # Additional indexers
        self.loc = LocIndexer(dataset=self)

    def __getitem__(self, cols):
        """Will return the column much like pandas.

        Args:
            cols (list/str): A list of columns or a column name string.

        Example:

        Returns:
            torch.Tensor: Tensor corresponding to the chosen columns.
        """
        return self.data[:, :, self._col_indexer(cols)]

    def __setitem__(self, key, item):
        """
        Args:
            key (list/str): Columns to overwrite.
            item (torch.Tensor): Data to overwrite with.

        Returns:
            None
        """
        if key in self.columns:
            self.data[:, :, self._col_indexer(key)] = item
        else:
            self.add_features(item, [key])

    def __len__(self):
        return self.data.size(0)

    def _col_indexer(self, cols):
        """ Returns a boolean list marking the index of the columns in the full column list. """
        return index_getter(self.columns, cols)

    def ragged_size(self, idx=None):
        """ Returns the total 'ragged' size. The time dimension returned is the sum of all lengths. """
        size = (self.size(0), sum(self.lengths), self.size(2))
        out = size if idx is None else size[idx]
        return out

    def init_assertions(self):
        """ Assertions that can be run to ensure class variables have matching sizes. """
        N, _, C = self.data.size()

        to_assert = [
            C == len(self.columns)
        ]

        assert all(to_assert), 'Sizes mismatch!'

    def add_features(self, data, columns=None):
        """Method for adding newly computed features to each column.

        If new features are wanted to be added to

        Args:
            data (torch.Tensor): Tensor of shape [N, L, C_new] where C_new are the new feature channels to be added.
            columns (list): List containing column names for each of the new features. If unspecified is filled as
                increasing integers.

        Returns:
            None
        """
        new_features = data.shape[2]

        # Logic if columns unspecified
        if columns is None:
            int_cols = [x for x in self.columns if isinstance(x, int)]
            if len(int_cols) == 0:
                int_cols = [-1]
            max_int = max(int_cols)
            columns = [str(x) for x in range(max_int + 1, max_int + 1 + new_features)]

        # Error handling
        assert data.shape[0:2] == self.data.shape[0:2], 'Dataset data and input data are different shapes.'
        assert len(columns) == new_features, 'Input data has a different length to input columns.'
        assert isinstance(columns, list), 'Columns must be inserted as list type.'
        assert len(set(columns)) == len(columns), 'Column names are not unique.'

        # Update
        self.data = torch.cat((self.data, data), dim=2)
        self.columns.extend(columns)

    def size(self, *args):
        return self.data.size(*args)

    def shape(self, *args):
        return self.data.shape(*args)

    def save(self, loc):
        """ Saves data, lengths and columns as a pickle. """
        items = self.data, self.lengths, self.columns
        save_pickle(items, loc)

    def load(self, loc):
        """ Reloads data, lengths and columns. """
        data, lengths, columns = load_pickle(loc)
        self.__init__(data, columns)
        self.lengths = lengths
        return self

    def to_list(self):
        """ Converts the tensor data back onto original length list format. """
        tensor_list = []
        for i, l in enumerate(self.lengths):
            tensor_list.append(self.data[i, 0:l, :])
        return tensor_list

    def to_ml(self):
        """ Converts onto a single tensor of original lengths. """
        return torch.cat(self.to_list())

    def subset(self, columns):
        """ Return only a subset of the dataset columns but as a TimeSeriesDataset. """
        assert isinstance(columns, list)
        assert all([x in self.columns for x in columns])
        dataset = TimeSeriesDataset(self[columns], columns)
        dataset.lengths = self.lengths
        return dataset


class LocIndexer():
    """ Emulates the pandas loc behaviour to work with TimeSeriesDataset. """
    def __init__(self, dataset):
        """
        Args:
            dataset (class): A TimeSeriesDataset class instance.
        """
        self.dataset = dataset

    def _tuple_loc(self, query):
        """ Loc getter if query is specified as an (id, column) tuple. """
        idx, cols = query

        # Handle single column case
        if isinstance(cols, str):
            cols = [cols]

        # Ensure cols exist and get integer locations
        col_mask = self.dataset._col_indexer(cols)

        if not isinstance(idx, slice):
            assert isinstance(idx, int), 'Either index with a slice (a:b) or an integer.'
            idx = slice(idx, idx+1)

        return self.dataset.data[idx, :, col_mask]

    def __getitem__(self, query):
        """
        Args:
            query [slice, list]: Works like the loc indexer, e.g. [1:5, ['col1', 'col2']].
        """
        if isinstance(query, tuple):
            output = self._tuple_loc(query)
        else:
            output = self.dataset.data[query, :, :]
        return output


def index_getter(full_list, idx_items):
    """Boolean mask for the location of the idx_items inside the full list.

    Args:
        full_list (list): A full list of items.
        idx_items (list/str): List of items you want the indexes of.

    Returns:
        list: Boolean list with True at the specified column locations.
    """
    # Turn strings to list format
    if isinstance(idx_items, str):
        idx_items = [idx_items]

    # Check that idx_items exist in full_list
    diff_cols = [c for c in idx_items if c not in full_list]
    assert len(diff_cols) == 0, "The following cols do not exist in the dataset: {}".format(diff_cols)

    # Actual masking
    col_idxs = [i for i, c in enumerate(full_list) if c in idx_items]

    return col_idxs


def torch_ffill(data):
    """ Forward fill for a torch tensor.

    This (currently) assumes a torch tensor input of shape [N, L, C] and will forward will along the 2nd (L)
    dimension.

    https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    """
    def ffill2d(arr):
        """ 2d ffill. """
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:, None], idx]
        return out

    data_ffilled = torch.Tensor([ffill2d(x.numpy().T) for x in data]).transpose(1, 2)
    return data_ffilled


# Features exported
def shock_index(data):
    """ HR/SBP ratio. """
    return data['HR'] / data['SBP']


def partial_sofa(data):
    """ Partial reconstruction of the SOFA score from features available in the sepsis dataset. """
    # Init the tensor
    N, L, C = data.data.size()
    sofa = torch.zeros(N, L, 1)

    # Coagulation
    platelets = data['Platelets']
    sofa[platelets >= 150] += 0
    sofa[(100 <= platelets) & (platelets < 150)] += 1
    sofa[(50 <= platelets) & (platelets < 100)] += 2
    sofa[(20 <= platelets) & (platelets < 50)] += 3
    sofa[platelets < 20] += 4

    # Liver
    bilirubin = data['Bilirubin_total']
    sofa[bilirubin < 1.2] += 0
    sofa[(1.2 <= bilirubin) & (bilirubin <= 1.9)] += 1
    sofa[(1.9 < bilirubin) & (bilirubin <= 5.9)] += 2
    sofa[(5.9 < bilirubin) & (bilirubin <= 11.9)] += 3
    sofa[bilirubin > 11.9] += 4

    # Cardiovascular
    map = data['MAP']
    sofa[map >= 70] += 0
    sofa[map < 70] += 1

    # Creatinine
    creatinine = data['Creatinine']
    sofa[creatinine < 1.2] += 0
    sofa[(1.2 <= creatinine) & (creatinine <= 1.9)] += 1
    sofa[(1.9 < creatinine) & (creatinine <= 3.4)] += 2
    sofa[(3.4 < creatinine) & (creatinine <= 4.9)] += 3
    sofa[creatinine > 4.9] += 4

    return sofa


"""
rolling.py
=============================
For calculating features over rolling windows.
"""
import torch
import warnings
import time


def timeit(method):
    """ Get the time it takes for a method to run.

    Args:
        method (function): The function to time.

    Returns:
        Method wrapped with an operation to time it.
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r \n  %2.2f ms' % (method, (te - ts) * 1000))
        return result
    return timed


def pytorch_rolling(x, dimension, window_size, step_size=1, return_same_size=True):
    """ Outputs an expanded tensor to perform rolling window operations on a pytorch tensor.

    Given an input tensor of shape [N, L, C] and a window length W, computes an output tensor of shape [N, L, C, W]
    where the final dimension contains the values from the current timestep to timestep - W + 1.

    Args:
        x (torch.Tensor): Tensor of shape [N, L, C].
        dimension (int): Dimension to open.
        window_size (int): Length of the rolling window.
        step_size (int): Window step, defaults to 1.
        return_same_size (bool): Set True to return a tensor of the same size as the input tensor with nan values filled
            where insufficient prior window lengths existed. Otherwise returns a reduced size
            tensor from the paths that had sufficient data.

    Returns:
        torch.Tensor: Tensor of shape [N, L, C, W] where the window values are opened into the fourth W dimension.
    """
    if return_same_size:
        x_dims = list(x.size())
        x_dims[dimension] = window_size - 1
        nans = np.nan * torch.zeros(x_dims)
        x = torch.cat((nans, x), dim=dimension)

    # Unfold ready for mean calculations
    unfolded = x.unfold(dimension, window_size, step_size)

    return unfolded


class RollingStatistic:
    """Applies statistics to a rolling window along the time dimension of a tensor.

    Given an input tensor of shape [N, L, C] and a specified window size, W, this function first expands the tensor to
    one of shape [N, L, C, W] where W has expanded out the time dimension (this here is the L-dimension). The final
    dimension contains the most recent W time-steps (with nans if not filled). The specified statistic is then computed
    along this W dimension to give the statistic over the rolling window.

    Example:
        >>> means = RollingStatistic(statistic='mean', window_length=5).transform(data)
    """
    def __init__(self, statistic, window_length, step_size=1, func_kwargs={}):
        """
        # TODO implement a method that removes statistics that contained insufficient data.
        Args:
            statistic (str): The statistic to compute.
            window_length (int): Length of the window.
            step_size (int): Window step size.
        """
        self.statistic = statistic
        self.window_length = window_length
        self.step_size = step_size
        self.func_kwargs = func_kwargs

    @staticmethod
    def count(data):
        counts = (~torch.isnan(data)).sum(axis=-1)
        return counts.to(data.dtype)

    @staticmethod
    def max(data):
        return data.max(axis=3)[0]
        # return torch.Tensor(np.nanmax(data, axis=3))

    @staticmethod
    def min(data):
        return data.min(axis=3)[0]
        # return torch.Tensor(np.nanmin(data, axis=3))

    @staticmethod
    def mean(data):
        return torch.Tensor(np.nanmean(data, axis=3))

    @staticmethod
    def var(data):
        return torch.Tensor(np.nanvar(data, axis=3))

    @staticmethod
    def change(data):
        """ Notes the change in the variable over the interval. """
        return data[:, :, :, -1] - data[:, :, :, 0]

    @staticmethod
    def moments(data, n=3):
        """Gets statistical moments from the data.

        Args:
            data (torch.Tensor): Pytorch rolling window data.
            n (int): Moments to compute up to. Must be >=2 computes moments [2, 3, ..., n].
        """
        # Removes the mean of empty slice warning
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        assert n >= 2, "Number of moments is {}, must be >= 2.".format(n)

        # Pre computation
        nanmean = torch.Tensor(np.nanmean(data, axis=3)).unsqueeze(-1)
        # frac = torch.Tensor(1 / (data.size(3) - np.isnan(data.numpy()).sum(axis=3)))
        frac = torch.Tensor(1 / (data.size(3) - np.isnan(data.numpy()).sum(axis=3) - 1))
        frac[(frac == float("Inf")) | (frac < 0)] = float('nan')
        mean_reduced = data - nanmean

        # Compute each moment individually
        moments = []
        for i in range(2, n+1):
            moment = torch.mul(frac, torch.Tensor((mean_reduced ** i).sum(axis=3)))
            moments.append(moment)
        moments = np.concatenate(moments, axis=2)
        moments = torch.Tensor(moments)

        return moments

    @timeit
    def transform(self, data):
        # Remove mean of empty slice warning
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Error handling
        assert self.statistic in dir(self), 'Statistic {} is not implemented via this method.'.format(self.statistic)

        # Setup function
        func = eval('self.{}'.format(self.statistic))

        # Make rolling
        rolling = pytorch_rolling(data, 1, self.window_length, self.step_size)

        # Apply and output
        output = func(rolling, **self.func_kwargs)

        return output

