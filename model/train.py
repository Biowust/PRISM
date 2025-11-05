import time
import numpy as np
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from sklearn.cluster import KMeans
from model2 import PRISM_module
from tqdm import tqdm


def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn


def gcn_loss(preds, labels, mu, logvar, n_nodes, norm):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


class PRISM:
    def __init__(
            self,
            X1,
            X2,
            graph_dict,
            rec_w=10,
            gcn_w=0.1,
            self_w=1,
            dec_kl_w=1,
            device = "cuda:0",
    ):

        self.rec_w = rec_w
        self.gcn_w = gcn_w
        self.self_w = self_w
        self.dec_kl_w = dec_kl_w
        self.device = device
  
       
        self.cell_num = len(X1)

        self.X1 = torch.FloatTensor(X1.copy()).to(self.device)
        self.X2 = torch.FloatTensor(X2.copy()).to(self.device)
        self.input1_dim = self.X1.shape[1]
        self.input2_dim = self.X2.shape[1]
        

        self.adj_norm = graph_dict["adj_norm"].to(self.device)
        self.adj_label = graph_dict["adj_label"].to(self.device)

        self.norm_value = graph_dict["norm_value"]

        
        self.model = PRISM_module(self.input1_dim, self.input2_dim).to(self.device)


   
    def train_without_dec(
            self,
            epochs=2000,
            lr=0.0012,
            decay=0.01,
            gradient_clipping=5.,
            N=1,
    ):
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=lr,
            weight_decay=decay)

        self.model.train()

 
        for _ in tqdm(range(epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            z1, mu1, logvar1, de_feat1, z2, mu2, logvar2, de_feat2, q, latent_z, loss_self = self.model(self.X1, self.X2, self.adj_norm)

            loss_gcn1 = gcn_loss(
                preds=self.model.dc1(z1),
                labels=self.adj_label.to_dense(),
                mu=mu1,
                logvar=logvar1,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )
            loss_gcn2 = gcn_loss(
                preds=self.model.dc2(z2),
                labels=self.adj_label.to_dense(),
                mu=mu2,
                logvar=logvar2,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )
            loss_gcn = loss_gcn1 + loss_gcn2

            
            loss_rec = reconstruction_loss(de_feat1, self.X1) + reconstruction_loss(de_feat2, self.X2)
            loss = self.rec_w * loss_rec + self.gcn_w * loss_gcn  
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clipping)
            self.optimizer.step()

        #     list_rec.append(loss_rec.detach().cpu().numpy())
        #     list_gcn.append(loss_gcn.detach().cpu().numpy())
        #     list_self.append(loss_self.detach().cpu().numpy())
        #
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(list_rec, label='rec')
        # ax.plot(list_gcn, label='gcn')
        # ax.plot(list_self, label='self')
        # ax.legend()
        # plt.show()


    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def process(self):
        self.model.eval()
        _, _, _, _, _, _, _, _, q, latent_z,_  = self.model(self.X1, self.X2, self.adj_norm)
        
        latent_z = latent_z.data.cpu().numpy()
        q = q.data.cpu().numpy()
        
        return latent_z, q



    def train_with_dec(
            self,
            epochs=2000,
            dec_interval=20,
            dec_tol=0.00,
            gradient_clipping=5.,
            N=1,
    ):
        # initialize cluster parameter
        # self.train_without_dec(
        #     epochs=epochs,
        #     lr=lr,
        #     decay=decay,
        #     N=N,
        # )
        self.train_without_dec()

        kmeans = KMeans(n_clusters=self.model.dec_cluster_n, n_init=self.model.dec_cluster_n * 2, random_state=42)
        test_z, _ = self.process()
        y_pred_last = np.copy(kmeans.fit_predict(test_z))

        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.model.train()

        for epoch_id in tqdm(range(epochs)):
            # DEC clustering update
            if epoch_id % dec_interval == 0:
                _, tmp_q = self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q))
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                self.model.train()
                if epoch_id > 0 and delta_label < dec_tol:
                    print('delta_label {:.4}'.format(delta_label), '< tol', dec_tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # training model
            torch.set_grad_enabled(True)
            z1, mu1, logvar1, de_feat1, z2, mu2, logvar2, de_feat2, out_q, latent_z, _ = self.model(self.X1, self.X2, self.adj_norm)
            

            loss_gcn1 = gcn_loss(
                preds=self.model.dc1(z1),
                labels=self.adj_label.to_dense(),
                mu=mu1,
                logvar=logvar1,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )
            loss_gcn2 = gcn_loss(
                preds=self.model.dc2(z2),
                labels=self.adj_label.to_dense(),
                mu=mu2,
                logvar=logvar2,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )
            loss_gcn = loss_gcn1 + loss_gcn2

        
            loss_rec = reconstruction_loss(de_feat1, self.X1) + reconstruction_loss(de_feat2, self.X2)

            
            # clustering KL loss
            loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
            loss = self.gcn_w * loss_gcn + self.dec_kl_w * loss_kl + self.rec_w * loss_rec
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clipping)
            self.optimizer.step()
