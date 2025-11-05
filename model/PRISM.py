import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from functools import partial


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = adj.matmul(support)
        output = self.act(output)
        return output
    

class AttentionLayer(nn.Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.alpha, -1))
    
        return torch.squeeze(emb_combined)
    

class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t())) 
        return adj
    

class PRISM_module(nn.Module):
    def __init__(
            self,
            input1_dim,
            input2_dim,
            gcn1_hidden1=64,
            gcn1_hidden2=16,
            gcn2_hidden1=64,
            gcn2_hidden2=16,
            p_drop=0.2,
            alpha=1.0,
            dec_clsuter_n=10,
    ):
        super(PRISM_module, self).__init__()
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.gcn1_hidden1 = gcn1_hidden1
        self.gcn1_hidden2 = gcn1_hidden2
        self.gcn2_hidden1 = gcn2_hidden1
        self.gcn2_hidden2 = gcn2_hidden2
        self.p_drop = p_drop
        self.alpha = alpha
        self.dec_cluster_n = dec_clsuter_n
        self.latent1_dim = self.gcn1_hidden2
        self.latent2_dim = self.gcn2_hidden2 
        
        
        # GCN1 layers
        self.gc1 = GraphConvolution(self.input1_dim, self.gcn1_hidden1, self.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(self.gcn1_hidden1, self.gcn1_hidden2, self.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(self.gcn1_hidden1, self.gcn1_hidden2, self.p_drop, act=lambda x: x)
        self.dc1 = InnerProductDecoder(self.p_drop, act=lambda x: x)


        # GCN2 layers
        self.gc4 = GraphConvolution(self.input2_dim, self.gcn2_hidden1, self.p_drop, act=F.relu)
        self.gc5 = GraphConvolution(self.gcn2_hidden1, self.gcn2_hidden2, self.p_drop, act=lambda x: x)
        self.gc6 = GraphConvolution(self.gcn2_hidden1, self.gcn2_hidden2, self.p_drop, act=lambda x: x)
        self.dc2 = InnerProductDecoder(self.p_drop, act=lambda x: x)

        self.decoder1 = GraphConvolution(self.latent1_dim, self.input1_dim, self.p_drop, act=lambda x: x)
        self.decoder2 = GraphConvolution(self.latent2_dim, self.input2_dim, self.p_drop, act=lambda x: x)

        self.atten_z = AttentionLayer(gcn1_hidden2, gcn2_hidden2)

        
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.gcn2_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


        self.criterion = self.setup_loss_fn(loss_fn='sce')

    def setup_loss_fn(self, loss_fn, alpha_l=3):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=3)
        else:
            raise NotImplementedError
        return criterion

    def encode1(self, x1, adj):
        hidden1 = self.gc1(x1, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def encode2(self, x2, adj):
        hidden2 = self.gc4(x2, adj)
        return self.gc5(hidden2, adj), self.gc6(hidden2, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    

    def forward(self, x1, x2, adj):

        mu1, logvar1 = self.encode1(x1, adj)
        mu2, logvar2 = self.encode2(x2, adj)
        
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)

        de_feat1 = self.decoder1(z1, adj)
        de_feat2 = self.decoder2(z2, adj)

        z = self.atten_z(z1, z2)
        

        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()


        # self-construction loss
        recon1 = de_feat1.clone()
        recon2 = de_feat2.clone()
        loss = self.criterion(recon1, x1) + self.criterion(recon2, x2)
        # loss = 0

        return z1, mu1, logvar1, de_feat1, z2, mu2, logvar2, de_feat2, q, z, loss
