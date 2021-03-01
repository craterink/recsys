from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class MovieLensDatasetHelper():
    cached_data = {}
    USER_IDX = 'userId'
    ITEM_IDX = 'movieId'
    RATING_IDX = 'rating'
    TIMESTAMP_IDX = 'timestamp'
    COLUMNS = [USER_IDX, ITEM_IDX, RATING_IDX, TIMESTAMP_IDX]
    ITEM_GENRES = ['Action', 'Adventure', 'Animation',
              'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western']

    @staticmethod
    def loadTVTSplitsFromFile(udata_file, val_split_perc=.1, test_split_perc=.1):
        # check in-memory cache
        if (udata_file, val_split_perc, test_split_perc) in MovieLensDatasetHelper.cached_data:
            return MovieLensDatasetHelper.cached_data[(udata_file, val_split_perc, test_split_perc)]

        # load and split data
        # split train/val/test *within* user by least-to-most recent
        # that is, we *won't* have to predict on new users
        df = pd.read_csv(udata_file, delimiter='\t', header=None, names=MovieLensDatasetHelper.COLUMNS)
        
        keep_users = df[MovieLensDatasetHelper.USER_IDX].unique()
        avg_inter_per_user = df.groupby(MovieLensDatasetHelper.USER_IDX).apply(len).mean()
        num_users = len(keep_users)
        unique_items = df[MovieLensDatasetHelper.ITEM_IDX].unique()
        num_items = len(unique_items)
        print('Num users = {}, num items = {}. Calculated {} average interactions per user.'.format(num_users, num_items, avg_inter_per_user))
        kval, ktest = int(val_split_perc*avg_inter_per_user), int(test_split_perc*avg_inter_per_user)
        print('Witholding {} validation interactions, {} test interactions.'.format(kval, ktest))
        sorted_groups = df.groupby(MovieLensDatasetHelper.USER_IDX).apply(lambda x: x.sort_values(by=MovieLensDatasetHelper.TIMESTAMP_IDX))
        train = sorted_groups.apply(lambda x: x.iloc[:max(len(x) - kval - ktest, 0)])
        val = df.groupby(MovieLensDatasetHelper.USER_IDX).apply(lambda x: x.iloc[max(len(x) - kval - ktest, 0):max(len(x) - ktest, 0)])
        test = df.groupby(MovieLensDatasetHelper.USER_IDX).apply(lambda x: x.iloc[max(len(x) - ktest, 0):])
        test = df.groupby(MovieLensDatasetHelper.USER_IDX).apply(lambda x: x.iloc[max(len(x) - ktest, 0):])
        print('Shapes: train {}; val {}; test {};'.format(train.shape, val.shape, test.shape))
        
        # update in-memory cache
        MovieLensDatasetHelper.cached_data[(udata_file, val_split_perc, test_split_perc)] = train, val, test, keep_users, unique_items, num_users, num_items
        
        return train, val, keep_users, unique_items, num_users, num_items

    @staticmethod
    def form_implicit_dataset(data):
        data = data.drop([MovieLensDatasetHelper.TIMESTAMP_IDX], axis=1)
        data[MovieLensDatasetHelper.RATING_IDX] = 1 # positive indicates interaction
        return data

    @staticmethod
    def negatively_sample(positives, neg_per_pos=4):
        all_items = positives[MovieLensDatasetHelper.ITEM_IDX].unique()
        def neg_sample_helper(user_positives):
            possible_negatives = np.setdiff1d(all_items, user_positives[MovieLensDatasetHelper.ITEM_IDX])
            negatives = np.random.choice(possible_negatives, size=min(len(possible_negatives), neg_per_pos*len(user_positives)), replace=False)
            negatives_df = pd.DataFrame().reindex(columns=user_positives.columns)
            negatives_df[MovieLensDatasetHelper.ITEM_IDX] = negatives
            negatives_df[MovieLensDatasetHelper.USER_IDX] = user_positives[MovieLensDatasetHelper.USER_IDX].values[0]
            negatives_df[MovieLensDatasetHelper.RATING_IDX] = 0
            return pd.concat([user_positives, negatives_df])
        # add NEG_PER_POS negative examples per row of implicit positives
        all_samples = positives.reset_index(level=0, drop=True).groupby([MovieLensDatasetHelper.USER_IDX]).apply(neg_sample_helper) # positives are already groupby'ed
        return all_samples

class MovieLens100KWithGenresDataset(Dataset):
    def __init__(self, udata_file, genres_data_file, split='train', neg_per_pos=4, tf_idf=False):
        self.data = MovieLens100KImplicitDataset(udata_file, split='train', neg_per_pos=neg_per_pos)
        self.genres_dataset = MovieLens100KGenresDataset(genres_data_file, tf_idf=tf_idf)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datapoint = self.data[idx]
        item = datapoint[1] 
        genres_data = self.genres_dataset[item - 1] # 1-indexed to 0-indexed
        return datapoint, genres_data

    def get_user_item_info(self):
        return self.data.num_users, self.data.num_items, self.genres_dataset.NUM_GENRES, self.data.user_idx_lookup, self.data.item_idx_lookup

class MovieLens100KGenresDataset(Dataset):
    GENRES = ['UNK', 'Action', 'Adventure', 'Animation',
              'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
              'Thriller', 'War', 'Western']
    NUM_GENRES = len(GENRES)
    def __init__(self, genres_file, tf_idf=False):
        # load data into pandas DataFrame from genre file
        data = np.loadtxt(genres_file, delimiter='|', encoding='latin-1', dtype=str)
        genres = data[:, -self.NUM_GENRES:].astype(float)
        
        # convert to TV-IDF if specified
        if tf_idf:
            N = genres.shape[0]
            TF_ij = genres # no normalization needed, as genres is binary vector (does not count per-doc frequency)
            IDF_i = np.log(N/genres.sum(axis=0))
            TF_IDF = TF_ij*np.expand_dims(IDF_i, 0)
            genres = TF_IDF

        self.genres_df = pd.DataFrame(data=genres, columns=self.GENRES)

    def __len__(self):
        return len(self.genres_df)
    
    def __getitem__(self, idx):
        return self.genres_df.iloc[idx].to_numpy()

class MovieLens100KImplicitDataset(Dataset):
    def __init__(self, udata_file, split='train', neg_per_pos=4):
        tvt_splits = MovieLensDatasetHelper.loadTVTSplitsFromFile(udata_file)
        self.num_users, self.num_items = tvt_splits[-2:]
        self.user_idxs = sorted(tvt_splits[-4])
        self.item_idxs = sorted(tvt_splits[-3])
        self.user_idx_lookup = np.zeros(max(self.user_idxs) + 1) # sparse vector where index i holds mapped embedding index for original index i
        self.item_idx_lookup = np.zeros(max(self.item_idxs) + 1)
        for enum_idx, idx in enumerate(self.user_idxs):
            self.user_idx_lookup[idx] = enum_idx
        for enum_idx, idx in enumerate(self.item_idxs):
            self.item_idx_lookup[idx] = enum_idx
        split_idx = ['train', 'val', 'test'].index(split)
        raw_data = tvt_splits[split_idx]
        implicit_data = MovieLensDatasetHelper.form_implicit_dataset(raw_data)
        self.data = MovieLensDatasetHelper.negatively_sample(implicit_data, neg_per_pos=neg_per_pos).to_numpy()
        # data is (Nx3), where N is number of interactions (pos and neg) and 3 is (userId, itemId, did_interact)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :]