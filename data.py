import numpy as np


class SessionDataset:
    """Credit to yhs-968/pyGRU4REC."""

    def __init__(self, data, session_key='SessionId', item_key='ItemId', time_key='Time'):
        self.df = data
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.idx2id = self.get_vocab()
        self.df['item_idx'] = self.df['ItemId'].map(self.idx2id.get)

        self.df.sort_values([session_key, time_key], inplace=True)
        self.click_offsets = self.get_click_offsets()
        self.session_idx_arr = np.arange(self.df[self.session_key].nunique())  # indexing to SessionId

    def get_vocab(self):
        return {index: item_id for item_id, index in enumerate(self.df['ItemId'].unique())}

    def get_click_offsets(self):
        """
        Return the indexes of the first click of each session IDs,
        """
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32)
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()
        return offsets


class SessionDataLoader:
    """Credit to yhs-968/pyGRU4REC."""

    def __init__(self, dataset: SessionDataset, batch_size=50):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_items = dataset.df['ItemId'].nunique() + 1
        self.click_offsets = self.dataset.click_offsets
        self.session_idx_arr = self.dataset.session_idx_arr
        self.done_sessions_counter = 0
        self.max_iter = 0

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.
        Yields:
            input (B,):  Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """

        df = self.dataset.df
        iters = np.arange(self.batch_size)
        max_iter = iters.max()
        start = self.click_offsets[self.session_idx_arr[iters]]  # Session Start
        end = self.click_offsets[self.session_idx_arr[iters] + 1]  # Session End
        mask = []  # indicator for the sessions to be terminated
        finished = False

        while not finished:
            min_len = (end - start).min()  # Shortest Session
            # Item indices (for embedding) for clicks where the first sessions start
            for i in range(min_len - 1):
                # Build inputs & targets
                inp = df['item_idx'].values[start + i]
                target = df['item_idx'].values[start + i + 1]
                yield inp, target, mask

            start, end, max_iter, finished = self.update_status(start, end, min_len, max_iter, finished)

    def update_status(self, start, end, min_len, max_iter, finished):
        # click indices where a particular session meets second-to-last element
        start = start + (min_len - 1)
        # see if how many sessions should terminate
        mask = np.arange(self.batch_size)[(end - start) <= 1]
        self.done_sessions_counter = len(mask)
        for idx in mask:
            max_iter += 1
            if max_iter >= len(self.click_offsets) - 1:
                self.max_iter = max_iter
                finished = True
                break
            # update the next starting/ending point
            start[idx] = self.click_offsets[self.session_idx_arr[max_iter]]
            end[idx] = self.click_offsets[self.session_idx_arr[max_iter] + 1]
        return start, end, max_iter, finished
