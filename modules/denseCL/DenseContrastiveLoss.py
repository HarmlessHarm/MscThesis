
import torch
import torch.nn.functional as F

class DenseContrastiveLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, dense_img, dense_pos, dense_neg):
        
        def reshape_bottleneck(x):
            return x.reshape(x.shape[0], x.shape[1], -1)
    

        dense_img = reshape_bottleneck(dense_img)
        dense_pos = reshape_bottleneck(dense_pos)
        dense_neg = reshape_bottleneck(dense_neg)
        
        assert dense_img.shape == dense_pos.shape and dense_img.shape == dense_neg.shape, "input shapes should be the same"
        
        B = dense_img.shape[0]
        D = dense_img.shape[1]
        S = dense_img.shape[2]
        
        temperature = 50
        
        loss_sum = 0
        for i in range(S):
            query = dense_img[:,:,i]

            # Find most similar target vector
            dist = F.cosine_similarity(dense_pos, query.unsqueeze(-1), dim=1)
            max_i = torch.argmax(dist, dim=1)
            batch_pos = torch.cat([ torch.index_select(a, 1, i).unsqueeze(0) for a, i in zip(dense_pos, max_i) ])
            
            dot_pos = torch.bmm(query.view(B,1,D), batch_pos.view(B,D,1))
            exp_pos = torch.exp(dot_pos / temperature)
            
            dot_neg = torch.bmm(query.view(B,1,D), dense_neg.view(B,D,S))
            exp_neg = torch.exp(dot_neg/temperature)
            sum_neg = torch.sum(exp_neg, axis=2, keepdim=True)
            
            loss_sum += -torch.log(exp_pos / (exp_pos + sum_neg))


        return torch.mean(loss_sum / S)