from torch.utils.data import Dataset

class BaseDataSet(Dataset):

    def __init__(self):
        super(Dataset, self).__init__()
        
    def name(self):
        return 'BaseDataSet'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
  
    def initialize(self, opt):
        pass

    def __len__(self):
        return 0
