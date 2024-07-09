import torch


class MetaPTP(torch.nn.Module):
    def __init__(self, horizon, ob_radius=2, hidden_dim=256):
        super().__init__()
        self.ob_radius = ob_radius
        self.horizon = horizon
        hidden_dim_fx = hidden_dim
        hidden_dim_fy = hidden_dim
        hidden_dim_by = 256
        feature_dim = 256
        self_embed_dim = hidden_dim
        neighbor_embed_dim = 128
        z_dim = 32
        d_dim = 2

        self.embed_size = hidden_dim
        self.heads = 1
        self.head_dim = self.embed_size // self.heads

        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(self_embed_dim, self_embed_dim*2),            
            torch.nn.ReLU6(),
            torch.nn.Linear(self_embed_dim*2, 12*20*2),
        )
        
        self.embed_s = torch.nn.Sequential(
            torch.nn.Linear(4, 64),             # v, a
            torch.nn.ReLU6(),
            torch.nn.Linear(64, self_embed_dim),
        )
        self.embed_n = torch.nn.Sequential(
            torch.nn.Linear(4, 64),             # dp, dv
            torch.nn.ReLU6(),
            torch.nn.Linear(64, neighbor_embed_dim),
            torch.nn.ReLU6(),
            torch.nn.Linear(neighbor_embed_dim, neighbor_embed_dim)
        )
        self.embed_k = torch.nn.Sequential(
            torch.nn.Linear(3, feature_dim),    # dist, bear angle, mpd
            torch.nn.ReLU6(),
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.ReLU6(),
            torch.nn.Linear(feature_dim, feature_dim)
        )
        self.embed_q = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim_fx, feature_dim),
            torch.nn.ReLU6(),
            torch.nn.Linear(feature_dim, feature_dim),
            torch.nn.ReLU6(),
            torch.nn.Linear(feature_dim, feature_dim)
        )
        self.attention_nonlinearity = torch.nn.LeakyReLU(0.2)

        self.rnn_fx = torch.nn.GRU(self_embed_dim+neighbor_embed_dim, hidden_dim_fx)
        self.rnn_fx_init = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim_fx), # dp
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_dim_fx, hidden_dim_fx*self.rnn_fx.num_layers),
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_dim_fx*self.rnn_fx.num_layers, hidden_dim_fx*self.rnn_fx.num_layers),
        )
        self.rnn_by = torch.nn.GRU(self_embed_dim+neighbor_embed_dim, hidden_dim_by)

        
        self.rnn_fy = torch.nn.GRU(z_dim, hidden_dim_fy)
        self.rnn_fy_init = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim_fx, hidden_dim_fy*self.rnn_fy.num_layers),
            torch.nn.ReLU6(),
            torch.nn.Linear(hidden_dim_fy*self.rnn_fy.num_layers, hidden_dim_fy*self.rnn_fy.num_layers)
        )

        self.values = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.fc_out = torch.nn.Linear(heads * self.head_dim, embed_size)
        self.wo_meta = torch.nn.Linear(2, 640, bias=False)

    def variance_loss(self, x):
        variance_sum = 0
        batch_size = len(x[0,:,0])
        for i in range(batch_size):
            variance_sum += torch.var(x[:, i, 0])  # 计算每个 batch 的方差之和
        return -variance_sum  # 负方差和

    def gaussian_loss(self, x):
        batch_size = len(x[0, :, 0])
        epsilon = 1e-6  # 添加一个小的常数来稳定计算
        loss_sum = 0
        for i in range(batch_size):
            batch_data = x[:, i, 0]
            mean =  torch.tensor(0)
            std =  torch.tensor(1.0) + epsilon  # 防止标准差为零
            # 高斯分布的负对数似然损失
            gaussian_nll = 0.5 * torch.log(2 * torch.pi * std ** 2) + 0.5 * ((batch_data - mean) ** 2) / (std ** 2)
            loss_sum += torch.mean(gaussian_nll)
            loss_g = loss_sum/batch_size
        return loss_g

    def var_loss(self, x):
        batch_size = len(x[0, :, 0])
        epsilon = 1e-6  # 添加一个小的常数来稳定计算
        loss_sum = 0
        for i in range(batch_size):
            batch_data = x[:, i, 0]
            sum = torch.sum(batch_data)

            loss_sum += torch.abs(sum-0.5)
            loss_g = loss_sum / batch_size
        return loss_g

    def linear_mapping(self, tensor):
        # 找到每个 [7, 512] 子张量的最小值和最大值
        min_vals = torch.min(tensor, dim=1, keepdim=True)[0]
        max_vals = torch.max(tensor, dim=1, keepdim=True)[0]

        # 计算归一化张量
        normalized_tensor = (tensor - min_vals) / (max_vals - min_vals + 1e-10)  # 加一个小值以防除以零
        return normalized_tensor


    
    def attention(self, q, k, mask):
        # q: N x d
        # k: N x Nn x d
        # mask: N x Nn
        e = (k @ q.unsqueeze(-1)).squeeze(-1)           # N x Nn
        e = self.attention_nonlinearity(e)              # N x Nn
        e[~mask] = -float("inf")
        att = torch.nn.functional.softmax(e, dim=-1)    # N x Nn
        return att.nan_to_num()
    
    def enc(self, x, neighbor, *, y=None):
        # x: (L1+1) x N x 6
        # y: L2 x N x 2
        # neighbor: (L1+L2+1) x N x Nn x 6
        with torch.no_grad():
            L1 = x.size(0)-1
            N = neighbor.size(1)
            Nn = neighbor.size(2)
            state = x

            x = state[...,:2]                       # (L1+1) x N x 2
            if y is not None:
                L2 = y.size(0)
                x = torch.cat((x, y), 0)            # (L+1) x N x 2
            else:
                L2 = 0
            
            v = x[1:] - x[:-1]                      # L x N x 2
            a = v[1:] - v[:-1]                      # (L-1) x N x 2
           
            a = torch.cat((state[1:2,...,4:6], a))  # L x N x 2

            neighbor_x = neighbor[...,:2]           # (L+1) x N x Nn x 2
            neighbor_v = neighbor[1:,...,2:4]       # L x N x Nn x 2
            
            dp = neighbor_x - x.unsqueeze(-2)       # (L+1) x N x Nn x 2
            dv = neighbor_v - v.unsqueeze(-2)       # L x N x Nn x 2

            # social features
            dist = dp.norm(dim=-1)                          # (L+1) x N x Nn
            mask = dist <= self.ob_radius
            dp0, mask0 = dp[0], mask[0]
            dp, mask = dp[1:], mask[1:]
            dist = dist[1:]
            dot_dp_v = (dp @ v.unsqueeze(-1)).squeeze(-1)   # L x N x Nn
            bearing = dot_dp_v / (dist*v.norm(dim=-1).unsqueeze(-1)) # L x N x Nn
            bearing = bearing.nan_to_num(0, 0, 0)
            dot_dp_dv = (dp.unsqueeze(-2) @ dv.unsqueeze(-1)).view(dp.size(0),N,Nn)
            tau = -dot_dp_dv / dv.norm(dim=-1)              # L x N x Nn
            tau = tau.nan_to_num(0, 0, 0).clip(0, 7)
            mpd = (dp + tau.unsqueeze(-1)*dv).norm(dim=-1)  # L x N x Nn
            features = torch.stack((dist, bearing, mpd), -1)# L x N x Nn x 3

        k = self.embed_k(features)                          # L x N x Nn x d
        s = self.embed_s(torch.cat((v, a), -1))
        n = self.embed_n(torch.cat((dp, dv), -1))           # L x N x Nn x ...

        ############ self-attention for meta weight #############
        values = s.permute(1,0,2) # (batch_size, value_len, embed_size)
        keys = s.permute(1,0,2)   # (batch_size, key_len, embed_size)
        query = s.permute(1,0,2)   # (batch_size, query_len, embed_size)
        mask_meta = None

        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask_meta is not None:
            energy = energy.masked_fill(mask_meta == 0, float("-1e20"))

        # attention_score = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        attention_score = energy.sum(-1) / (self.embed_size ** (1 / 2))
        attention_score = attention_score.permute(2, 0, 1)
        meta_weight = -attention_score

        h = self.rnn_fx_init(dp0)                           # N x Nn x d
        h = (mask0.unsqueeze(-1) * h).sum(-2)               # N x d
        h = h.view(N, -1, self.rnn_fx.num_layers)
        h = h.permute(2, 0, 1).contiguous()


        for t in range(L1):
            q = self.embed_q(h[-1])                         # N x d
            att = self.attention(q, k[t], mask[t])          # N x Nn
            x_t = att.unsqueeze(-2) @ n[t]                  # N x 1 x d
            x_t = x_t.squeeze(-2)                           # N x d
            x_t = torch.cat((x_t, -s[t]*meta_weight[t]), -1).unsqueeze(0)
            _, h = self.rnn_fx(x_t, h)


        x = h[-1]
        if y is None: return x
        mask_t = mask[L1:L1+L2].unsqueeze(-1)               # L2 x N x Nn x 1
        n_t = n[L1:L1+L2]                                   # L2 x N x Nn x d
        n_t = (mask_t * n_t).sum(-2)                        # L2 x N x d
        s_t = s[L1:L2+L2]
        x_t = torch.cat((n_t, s_t), -1)
        x_t = torch.flip(x_t, (0,))
        b, _ = self.rnn_by(x_t)                             # L2 x N x n_layer*d
        if self.rnn_by.num_layers > 1:
            b = b[...,-b.size(-1)//self.rnn_by.num_layers:]
        b = torch.flip(b, (0,))
        return x, b
    
    def forward(self, x,neighbor,y=None):

        self.rnn_fx.flatten_parameters()
        self.rnn_fy.flatten_parameters()
    
       
        
        batch_size = x.size(1)
        
        
        if neighbor is None:
            neighbor_shape = [_ for _ in x.shape]
            neighbor_shape.insert(-1, 0)
            neighbor = torch.empty(neighbor_shape, dtype=x.dtype, device=x.device)
        
        N = x.size(1)

        neighbor = neighbor[:x.size(0)]
        h = self.enc(x, neighbor)
        
        h = self.rnn_fy_init(h)
        h = h.view(N, -1, self.rnn_fy.num_layers)
        h = h.permute(2, 0, 1)

        pred_trajs = self.reg_head(h[-1]).reshape(batch_size, 20, 12, 2)
        pred_trajs = torch.cumsum(pred_trajs, dim=2)
        endpoint = x[-1,:,None,:2]
        
        pred_trajs = pred_trajs + endpoint[:,None,:,:]
        
        return pred_trajs
    
    def get_loss(self, pred_trajs, gt_trajs):
        gt_trajs = gt_trajs.permute(1,0,2)
        loss = {}
       
        err = torch.norm(pred_trajs - gt_trajs.unsqueeze(1), dim=-1)
        batch_ade = err.mean(dim=-1) # batch_size pred_num
        closest_indices = torch.min(batch_ade, dim=-1)[1]
        reg_loss = err[torch.LongTensor(range(closest_indices.shape[0])), closest_indices] # [batch_size]
        loss['loss'] = reg_loss.mean()
        
        return loss
