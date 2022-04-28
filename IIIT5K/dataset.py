from torch.utils.data import Dataset
import pickle
import string
import numpy as np

class IIIT5KDataset(Dataset):
    def __init__(self, split=None, directory='IIIT5K/dataset/'):
        super().__init__()
        imgs = []
        lbls = []
        bboxes = []
        
        if split == 'train':
            range_ = range(8)
        elif split == 'val':
            range_ = range(8, 9)
        elif split == 'test':
            range_ = range(9, 10)
        else:
            range_ = range(10)

        for i in range_:
            file_name = directory + f'shard_{i}.pkl'
            with open(file_name, 'rb') as file:
                temp_pickle = pickle.load(file)
            imgs += temp_pickle[0]
            lbls += temp_pickle[1]
            bboxes += temp_pickle[2]
            
        # Get max label size.
        max_label_size = 0
        for lbl in lbls:
            if len(lbl) > max_label_size:
                max_label_size = len(lbl)
            
        # Convert labels to character indexes in string.printable
        for i, lbl in enumerate(lbls):
            lbl = list(lbl)
            lbl = [string.printable.index(c) for c in lbl]
            lbl += [len(string.printable) for i in range(max_label_size - len(lbl) + 1)]  # eow
            # lbl = [np.eye(len(string.printable) + 1)[l] for l in lbl]
            
            lbls[i] = lbl

        self.images = imgs
        self.labels = lbls
        self.bbox = bboxes

    def __getitem__(self, index):
        img = self.images[index]
        lbl = self.labels[index]
        bbox = self.bbox[index]

        return (img, lbl)  # (img, lbl, bbox)

    def __len__(self):
        return len(self.bbox)  # Probably the most efficient len call