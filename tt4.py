
# 改进一下loss计算代码
# 矩阵加速
import torch 
import time
N = 235465
pt_offset_labels = torch.rand(N, 3) * 20
pos_inds = torch.arange(N)
pt_offsets = torch.rand(N, 3) * 21 - 0.5
pt_offset_labels2 = pt_offset_labels
pos_inds2 = pos_inds
pt_offsets2 = pt_offsets
# pt_offsets = torch.tensor([[1,2.3,3.5], [4.2, 5.1, 8], [6.3, 9.2, 10.5]], dtype=torch.float)
# pt_offset_labels: (N, 3) float, cuda
# # pt_offsets: (N, 3) float, cuda
begin = time.time()
pt_offset_labels_norm = torch.norm(pt_offset_labels[pos_inds], p=2, dim=1) #(n, )
pt_offset_labels_ = pt_offset_labels[pos_inds] / (pt_offset_labels_norm.unsqueeze(-1) + 1e-9) #(n, )     
pt_offsets_norm = torch.norm(pt_offsets[pos_inds], p=2, dim=1)
pt_offsets_ = pt_offsets[pos_inds] / (pt_offsets_norm.unsqueeze(-1) + 1e-9) #(n, )
direction_diff = -torch.sum(pt_offset_labels_ * pt_offsets_, dim=-1) #(n, )
direction_loss = torch.sum(direction_diff) / torch.sum(pos_inds)
end = time.time()
print(end - begin)
print(direction_loss)
# pt_offset_labels: (N, 3) float, cuda
# pt_offsets: (N, 3) float, cuda
#Compute the dot product between normalized pt_offset_labels and normalized pt_offsets 
pt_offset_labels_norm2 = torch.norm(pt_offset_labels2[pos_inds2], p=2, dim=1, keepdim=True)
pt_offsets_norm2 = torch.norm(pt_offsets2[pos_inds2], p=2, dim=1, keepdim=True)
dot_product = torch.sum(pt_offset_labels2[pos_inds2] * pt_offsets2[pos_inds2], dim=-1, keepdim=True)
direction_diff2 = -dot_product / (pt_offset_labels_norm2 * pt_offsets_norm2 + 1e-9)
# Compute the mean of direction_diff over positive indices
direction_loss2 = torch.sum(direction_diff2) / torch.sum(pos_inds2)
print(time.time() - end)
print(direction_loss2)
end = time.time()
