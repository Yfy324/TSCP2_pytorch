#Reference: https://theaisummer.com/simclr/
import torch
import torch.nn as nn
import torch.nn.functional as F

def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)

class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)  # L_p范数
        z_j = F.normalize(proj_2, p=2, dim=1)
        
        similarity_matrix = self.calc_similarity_batch(z_i, z_j)  # 有一个zi zj 沿着batch维度concat操作

        sim_ij = torch.diag(similarity_matrix, batch_size)  # 右上对角线
        sim_ji = torch.diag(similarity_matrix, -batch_size) # 左下对角线

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = device_as(self.mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))  # 没有用ce-loss，直接复现的公式
        loss = torch.sum(all_losses) / (2 * self.batch_size)
        
        mean_sim = similarity_matrix.mean()
        
        mean_neg = torch.mul(similarity_matrix, device_as(self.mask, similarity_matrix)).mean()  # x 逐元素相乘，只mask了主对角线元素，得到的是pos+neg!
        return loss, mean_sim, mean_neg
