import numpy as np


class SessionDataset:
    """Credit to yhs-968/pyGRU4REC."""

    def __init__(self, data, session_key='SessionId', item_key='ItemId', time_key='Time'):
        self.df = data
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.idx2id = self.add_item_indices()
        self.df.sort_values([session_key, time_key], inplace=True)
        # clicks within a session are next to each other, where the clicks within a session are time-ordered.
        self.click_offsets = self.get_click_offsets()
        self.session_idx_arr = np.arange(self.df[self.session_key].nunique())  # indexing to SessionId

    def add_item_indices(self):
        idx2id = {index: item_id for item_id, index in enumerate(self.df['ItemId'].unique())}
        self.df['item_idx'] = self.df['ItemId'].map(idx2id.get)
        return idx2id

    @property
    def items(self):
        return self.df['ItemId'].unique()

    def get_click_offsets(self):
        """
        Return the offsets of the beginning clicks of each session IDs,
        where the offset is calculated against the first click of the first session ID.
        """
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32)
        # group & sort the df by session_key and get the offset values
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()

        return offsets


class SessionDataLoader:
    """Credit to yhs-968/pyGRU4REC."""

    def __init__(self, dataset, batch_size=50):
        """
        A class for creating session-parallel mini-batches.
        Args:
            dataset (SessionDataset): the session dataset to generate the batches from
            batch_size (int): size of the batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.done_sessions_counter = 0

    def __iter__(self):  # https://dojang.io/mod/page/view.php?id=2405
        """ Returns the iterator for producing session-parallel training mini-batches.
        Yields:
            input (B,):  Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """

        df = self.dataset.df
        self.n_items = df['ItemId'].nunique() + 1
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = np.arange(self.batch_size)
        max_iter = iters.max()
        start = click_offsets[session_idx_arr[iters]]  # Session Start
        end = click_offsets[session_idx_arr[iters] + 1]  # Session End
        mask = []  # indicator for the sessions to be terminated
        finished = False

        while not finished:
            min_len = (end - start).min()  # Shortest Session
            # Item indices (for embedding) for clicks where the first sessions start
            for i in range(min_len - 1):
                # Build inputs & targets
                inp = df.item_idx.values[start + i]
                target = df.item_idx.values[start + i + 1]
                yield inp, target, mask

            # click indices where a particular session meets second-to-last element
            start = start + (min_len - 1)
            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end - start) <= 1]
            self.done_sessions_counter = len(mask)
            for idx in mask:
                max_iter += 1
                if max_iter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = max_iter
                start[idx] = click_offsets[session_idx_arr[max_iter]]
                end[idx] = click_offsets[session_idx_arr[max_iter] + 1]
