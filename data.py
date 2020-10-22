import numpy as np


class SessionDataLoader:
    """Credit to yhs-968/pyGRU4REC."""

    def __init__(self, data, batch_size=50):
        self.df = data
        self.idx2id = self.get_vocab()
        self.df['item_idx'] = self.df['ItemId'].map(self.idx2id.get)

        self.df.sort_values(['SessionId', 'Time'], inplace=True)
        self.click_offsets = self.get_click_offsets()
        self.session_idx_arr = np.arange(self.df['SessionId'].nunique())  # indexing to SessionId
        self.batch_size = batch_size

    def get_vocab(self):
        return {index: item_id for item_id, index in enumerate(self.df['ItemId'].unique())}

    def get_click_offsets(self):
        """
        Return the indexes of the first click of each session IDs,
        """
        offsets = np.zeros(self.df['SessionId'].nunique() + 1, dtype=np.int32)
        offsets[1:] = self.df.groupby('SessionId').size().cumsum()
        return offsets

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.
        Yields:
            input (B,):  Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """

        start, end, mask, last_session, finished = self.initialize()
        """
        start : Index Where Session Start
        end : Index Where Session End
        mask : indicator for the sessions to be terminated
        """

        while not finished:
            min_len = (end - start).min() - 1  # Shortest Length Among Sessions
            for i in range(min_len):
                # Build inputs & targets
                inp = self.df['item_idx'].values[start + i]
                target = self.df['item_idx'].values[start + i + 1]
                yield inp, target, mask

            start, end, mask, last_session, finished = self.update_status(start, end, min_len, last_session, finished)

    def initialize(self):
        first_iters = np.arange(self.batch_size)
        last_session = first_iters[-1]
        start = self.click_offsets[self.session_idx_arr[first_iters]]
        end = self.click_offsets[self.session_idx_arr[first_iters] + 1]
        mask = []
        finished = False
        return start, end, mask, last_session, finished

    def update_status(self, start, end, min_len, last_session, finished):
        start += min_len
        mask = np.arange(self.batch_size)[(end - start) == 1]

        for i, idx in enumerate(mask, start=1):
            new_session = last_session + i
            if new_session > self.session_idx_arr[-1]:
                finished = True
                break
            # update the next starting/ending point
            start[idx] = self.click_offsets[self.session_idx_arr[new_session]]
            end[idx] = self.click_offsets[self.session_idx_arr[new_session] + 1]

        last_session += len(mask)
        return start, end, mask, last_session, finished
