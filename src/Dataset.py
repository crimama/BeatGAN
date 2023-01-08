from torch.utils.data import Dataset,DataLoader


class SwatDataset(Dataset):
    def __init__(self,data,mode='train'):
        super(SwatDataset,self).__init__()
        self.train_x = data['train']['x'] 
        self.train_y = data['train']['y']
        self.test_x = data['test']['x'] 
        self.test_y = data['test']['y']
        self.mode = mode 

    def __len__(self):
        if self.mode == 'train':
            return self.train_x.shape[0]
        if self.mode == 'test':
            return self.test_x.shape[0]
        
    def __getitem__(self,idx):
        if self.mode == 'train':
            return self.train_x[idx],self.train_y[idx]
        else:
            return self.test_x[idx],self.test_y[idx]
        
  