
import torch
import torch.nn.functional as F

class GlobalContrastiveLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, global_img, global_pos, global_neg):
    
        assert global_img.shape == global_pos.shape and global_img.shape == global_neg.shape, "input shapes should be the same"
                
        B = global_img.shape[0]
        D = global_img.shape[1]
        
        temperature = 50
        
        query = global_img

        # Find most similar target vector
        dist = F.cosine_similarity(global_pos, query, dim=1)
        max_i = torch.argmax(dist, dim=1)
        batch_pos = torch.cat([ torch.index_select(a, 1, i).unsqueeze(0) for a, i in zip(global_pos, max_i) ])

        dot_pos = torch.bmm(query.view(B,1,D), batch_pos.view(B,D,1))
        exp_pos = torch.exp(dot_pos / temperature)

        dot_neg = torch.bmm(query.view(B,1,D), global_neg.view(B,D,1))
        exp_neg = torch.exp(dot_neg/temperature)
        sum_neg = torch.sum(exp_neg, axis=2, keepdim=True)

        global_loss = -torch.log(exp_pos / (exp_pos + sum_neg))

        return torch.mean(global_loss)
