import torch
import random
import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from scipy.stats import kde
import ipdb

def vis_distribution(p_z, save_dir,  Nn=1000, batch_size=128,h_dim=32, samples=None):
    
    z_ls = [p_z.sample() for _ in range(Nn)]
    if samples is not None:
        samples = samples.reshape(batch_size,-1,h_dim)
        sample_num = samples.size(1)
    zs = torch.cat(z_ls, dim=0).reshape(batch_size, -1, h_dim)
    
    pca = PCA(n_components=2)  # 将数据降维到二维
    
    
    # for batch_idx in range(batch_size):
    for batch_idx in range(10):
        # ipdb.set_trace()
        batch_data = zs[batch_idx]
        if samples is not None:
            
            batch_data = torch.cat([batch_data,samples[batch_idx]],dim=0)
        
        transformed_zs = pca.fit_transform(batch_data.cpu())
        if samples is not None:
            sample = transformed_zs[-sample_num:]
        k = kde.gaussian_kde(transformed_zs[:Nn].T)
        
        x, y = np.mgrid[-4:4:100j, -4:4:100j]
        positions = np.vstack([x.ravel(), y.ravel()])
        z = np.reshape(k(positions).T, x.shape)

        # 绘制热力图
        
        plt.imshow(z, origin='lower', aspect='auto', extent=[-4, 4, -4, 4], cmap='viridis')
        if samples is not None:
            plt.scatter(sample[...,0], sample[...,1], marker='x', color='r')
        plt.savefig(save_dir+str(batch_idx)+'.png')
        plt.close()
    
    
def vis(obs_trajs, queries=None, gt_trajs=None, y_tests =None,y_linear=None,y_kf =None, pred_trajs=None, global_fig=False, save_path=None):
    
    batch_size = obs_trajs.shape[0]

    if y_tests is not None:
        y_tests = y_tests.cpu().numpy()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if queries is not None:
        queries = queries.cpu().numpy()
    
    if gt_trajs is not None:
        gt_trajs = gt_trajs.cpu().numpy()
    
    if pred_trajs is not None:
        pred_trajs = pred_trajs.cpu().numpy()

    obs_trajs = obs_trajs.cpu().numpy()

    if global_fig:
        for i in range(batch_size):
            obs_traj = obs_trajs[i]
            obs_x = obs_traj[:, 0]
            obs_y = obs_traj[:, 1]
            plt.plot(obs_x, obs_y,  marker='.')
        plt.savefig('vis/train_global.png')
        print('global figure finished!')
    for i in range(batch_size):
        
        plt.clf()

        obs_traj = obs_trajs[i]
        obs_x = obs_traj[:, 0]
        obs_y = obs_traj[:, 1]
        plt.plot(obs_x, obs_y, c='green',  marker='.', label = 'obs')

       

        
        
        if pred_trajs is not None:
            if y_tests is not None:
                y_test = y_tests[i]
            pred_traj = pred_trajs[i]
            
            # ipdb.set_trace()
            if len(pred_traj.shape) == 3:
                
                plt.plot(pred_traj[0, :, 0], pred_traj[0, :, 1], c='orange', marker='.', alpha=0.5, label='pred')
                for num in range(pred_traj.shape[0]-1):
                    plt.plot(pred_traj[num+1, :, 0], pred_traj[num+1, :, 1], c='orange', marker='.', alpha=0.5)
                    # plt.plot(pred_traj[num+10, :3, 0], pred_traj[num+10, :3, 1], c='black', marker='.')
            # 
                
            #     plt.plot(pred_traj[0, :, 0], pred_traj[0, :, 1], c='orange', marker='.')
            #     plt.plot(pred_traj[1, :, 0], pred_traj[1, :, 1], c='red', marker='.')
            #     plt.plot(pred_traj[2, :, 0], pred_traj[2, :, 1], c='blue', marker='.')
            #     if y_tests is not None:
            #         # ipdb.set_trace()
            #         plt.plot(y_test[0, :, 0], y_test[0, :, 1], c='black', marker='.')
            #         plt.plot(y_test[1, :, 0], y_test[1, :, 1], c='black', marker='.')
            #         plt.plot(y_test[2, :, 0], y_test[2, :, 1], c='black', marker='.')
                            
            else:
                pred_x = pred_traj[:, 0]
                pred_y = pred_traj[:, 1]
                plt.plot(pred_x, pred_y, c='blue', marker='.', label='pred')

            if gt_trajs is not None:
                gt_traj = gt_trajs[i]
                gt_x = gt_traj[:, 0]
                gt_y = gt_traj[:, 1]
                plt.plot(gt_x, gt_y, c='red',  marker='.', label='gt')
        plt.legend()
        plt.savefig(save_path + '/fig_' + str(i) + '.png', dpi=300)
        plt.close()
    
    print('plotting finished!')
    return


def ADE_FDE(y_, y, batch_first=True):
    # average displacement error
    # final displacement error
    # y_, y: S x L x N x 2
    if torch.is_tensor(y):
        err = (y_ - y[:,None,:,:]).norm(dim=-1)
        
    else:
        err = np.linalg.norm(np.subtract(y_, y[:,None,:,:]), axis=-1)

    if len(err.shape) == 1:
        fde = err[-1]
        ade = err.mean()
    elif batch_first:
        fde = err[..., -1]
        ade = err.mean(-1)
    else:
        fde = err[..., -1, :]
        ade = err.mean(-2)
    return ade, fde

def kmeans(k, data, iters=None):
    centroids = data.copy()
    np.random.shuffle(centroids)
    centroids = centroids[:k]

    if iters is None: iters = 100000
    for _ in range(iters):
    # while True:
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        closest = np.argmin(distances, axis=0)
        centroids_ = []
        for k in range(len(centroids)):
            cand = data[closest==k]
            if len(cand) > 0:
                centroids_.append(cand.mean(axis=0))
            else:
                centroids_.append(data[np.random.randint(len(data))])
        centroids_ = np.array(centroids_)
        if np.linalg.norm(centroids_ - centroids) < 0.0001:
            break
        centroids = centroids_
    return centroids

def FPC(y, n_samples):
    # y: S x L x 2
    goal = y[...,-1,:2]
    goal_ = kmeans(n_samples, goal)
    dist = np.linalg.norm(goal_[:,np.newaxis,:2] - goal[np.newaxis,:,:2], axis=-1)
    chosen = np.argmin(dist, axis=1)
    return chosen
    
def seed(seed: int):
    rand = seed is None
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = not rand
    torch.backends.cudnn.benchmark = rand

def get_rng_state(device):
    return (
        torch.get_rng_state(), 
        torch.cuda.get_rng_state(device) if torch.cuda.is_available and "cuda" in str(device) else None,
        np.random.get_state(),
        random.getstate(),
        )

def set_rng_state(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None: torch.cuda.set_rng_state(state[1], device)
    np.random.set_state(state[2])
    random.setstate(state[3])
