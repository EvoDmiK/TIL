from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    
    def __init__(self, gamma = 0 , alpha = None, size_avg = True):
        
        super(FocalLoss, self).__init__()
        self.gamma    = gamma
        self.alpha    = alpha
        self.size_avg = size_avg
        
    
    def forward(self, input, target):
        
        if input.dim() > 2:
            
            ## 4차원 데이터를 2차원 데이터로 변경
            ## 1. B, C, H, W -> B, C, H * W
            input = input.view(input.size(0), input.size(1), -1)
            
            ## 2. B, C, H * W -> B, H*W, C
            input = input.transpose(1, 2)
            
            ## 3. B, H*W, C -> B*W*H, C
            input = input.contiguous().view(-1, input.size(2))
            
        target = target.view(-1, 1)
        log_pt = F.log_softmax(input)

        log_pt = log_pt.gather(1, target)
        log_pt = log_pt.view(-1)
        pt     = Variable(log_pt.data.exp())
        
        if isinstance(self.alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(self.alpha, list): self.alpha = torch.Tensor(self.alpha)
        
        if self.alpha is not None:
            
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
                
            at = self.alpha.gather(0, target.data.view(-1))
            log_pt = log_pt * Variable(at)
            
        loss = -1 * (1 - pt) ** self.gamma * log_pt
        return loss.mean() if self.size_avg else loss.sum()