import numpy as np

class Dataset():
    def __init__(self, x, ratio=None, y=None):
        self.x = x
        self.next_idx = 0
        self.next_idx_trn = 0
        self.next_idx_val = 0
        self.size = len(x)
        if ratio:
            thres = int(self.size * ratio)
            self.x_train = self.x[:thres]
            self.x_val = self.x[thres:]
            self.thres = thres
        
    def shuffle(rseed=None):
        if not rseed:
            np.random.shuffle(self.x)
        
    def _next_batch(self,dset,batch_size,roll_over=False):
        size = len(dset)
        if self.next_idx + batch_size >= size:
            if roll_over:
                batch = dset[self.next_idx:]
                left_over = batch_size - (size - self.next_idx)
                self.next_idx = left_over
                if type(batch) == 'list':
                    batch.extend(dset[:self.next_idx])
                else:
                    batch = np.concatenate((batch, dset[:self.next_idx]), axis=0)
                return batch
            else:
                old_idx = self.next_idx
                self.next_idx = 0
                return dset[old_idx:]
        else:
            old_idx = self.next_idx
            self.next_idx += batch_size
            return dset[old_idx:self.next_idx]
        
    def next_batch(self,batch_size,roll_over=False):
        self.next_idx = self.next_idx_trn
        batch = self._next_batch(self.x_train,batch_size,roll_over)
        self.next_idx_trn = self.next_idx
        return batch
    
    def validation_batch(self,batch_size='full',roll_over=False):
        if batch_size == 'full':
            self.next_idx = 0
            return self._next_batch(self.x_val,batch_size=len(self.x_val))
        else:
            self.next_idx = self.next_idx_val
            batch = self._next_batch(self.x_val,batch_size,roll_over)
            self.next_idx_val = self.next_idx
            return batch
