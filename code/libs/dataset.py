import random
import numpy as np
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from .audio import WaveReader

def make_dataloader(is_train,
                    mix_dir,
                    sample_rate: int,
                    batch_size: int,
                    num_workers: int,
                    chunk_size: int,
                    ):
    dataset = My_dataset(mix_dir, sample_rate)
    return My_dataLoader(dataset,
                      is_train,
                      batch_size,
                      num_workers,
                      chunk_size,
                      )
			    
class My_dataset(Dataset):
    """
    Per Utterance Loader
    """
    def __init__(self, mix_scp, sample_rate):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
            
    def __getitem__(self, index):
        key = self.mix.index_keys[index]
        mix = self.mix[key]
        return {
            "mix": mix.astype(np.float32),
			"key": key
        }
    def __len__(self):
        return len(self.mix)

class My_dataLoader(object):
   
    def __init__(self,
                 dataset,
                 train,
                 batch_size,
                 num_workers,
                 chunk_size,
                 ):
        self.train = train
        self.batch_size = batch_size
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                    )
        # just return batch of egs, support multiple workers
        self.eg_loader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    collate_fn=self._collate,
                                    num_workers=num_workers,
                                    shuffle=train,
                                    pin_memory=True
                                    )

    def _collate(self, batch):
        """
        Online split utterances
        """
        chunk = []
        for eg in batch:
            chunk += self.splitter.split(eg)
        return chunk
    
    def _merge(self, chunk_list):
        """
        Merge chunk list into mini-batch
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate((chunk_list[s:s + self.batch_size]))
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj

class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """
    def __init__(self, chunk_size, train=True):
        self.chunk_size = chunk_size
        self.train = train

    def _make_chunk(self, eg, s):
        """
        Make a chunk instance, which contains:
            "mix": ndarray,
            "ref": [ndarray...]
        """
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        return chunk

    def split(self, eg):
        N = eg["mix"].size
        chunks = []
        if N < self.chunk_size:
            units = self.chunk_size // N
            noisy_ds_final = []
            for i in range(units):
                noisy_ds_final.append(eg["mix"])
            noisy_ds_final.append(eg["mix"][: self.chunk_size%N])
            noisy_ds = np.concatenate(noisy_ds_final)
            chunk = dict()
            chunk["mix"] = noisy_ds
            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.chunk_size) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.chunk_size
        return chunks
