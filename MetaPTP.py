import os, sys, time
import importlib
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model import MetaPTP
from data import Dataloader
from utils import ADE_FDE, FPC, seed, get_rng_state, set_rng_state, vis

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs='+', default=[])
parser.add_argument("--test", nargs='+', default=[])
parser.add_argument("--frameskip", type=int, default=1)
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--device", type=str, default='0')
parser.add_argument("--seed", type=int, default=1)

if __name__ == "__main__":
    torch.manual_seed(3407)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(3407)
    
    settings = parser.parse_args()
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    settings.device = "cuda:" + settings.device
    settings.device = torch.device(settings.device)
    
    seed(settings.seed)
    init_rng_state = get_rng_state(settings.device)
    rng_state = init_rng_state

    ###############################################################################
    #####                                                                    ######
    ##### prepare datasets                                                   ######
    #####                                                                    ######
    ###############################################################################
    kwargs = dict(
            batch_first=False, frameskip=settings.frameskip,
            ob_horizon=config.OB_HORIZON, pred_horizon=config.PRED_HORIZON,
            device=settings.device, seed=settings.seed)
    train_data, test_data = None, None
    if settings.test:
        print(settings.test)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.test))]
        else:
            inclusive = None
        test_dataset = Dataloader(
            settings.test, **kwargs, inclusive_groups=inclusive,
            batch_size=config.BATCH_SIZE, shuffle=False
        )
        test_data = torch.utils.data.DataLoader(test_dataset, 
            collate_fn=test_dataset.collate_fn,
            batch_sampler=test_dataset.batch_sampler
        )
        def test(model):
            sys.stdout.write("\r\033[K Evaluating...{}/{}".format(
                0, len(test_dataset)
            ))
            tic = time.time()
            model.eval()
            ADE, FDE = [], []
            set_rng_state(init_rng_state, settings.device)
            x_ls, gt_ls, y_pred_ls = [], [], []
            with torch.no_grad():
                for x, y, neighbor in test_data:
                        
                    y_ = model(x, neighbor)
                    y = y.permute(1,0,2)
                    
                    ade, fde = ADE_FDE(y_, y)
                    
                    if config.PRED_SAMPLES > 0:
                        ade = torch.min(ade, dim=1)[0]
                        fde = torch.min(fde, dim=1)[0]
                    
                    x_ls.append(x)
                    gt_ls.append(y)
                    y_pred_ls.append(y_)
                    ADE.append(ade)
                    FDE.append(fde)
                    
            
            ADE = torch.cat(ADE)
            FDE = torch.cat(FDE)
            if torch.is_tensor(config.WORLD_SCALE) or config.WORLD_SCALE != 1:
                if not torch.is_tensor(config.WORLD_SCALE):
                    config.WORLD_SCALE = torch.as_tensor(config.WORLD_SCALE, device=ADE.device, dtype=ADE.dtype)
                
                ADE *= config.WORLD_SCALE
                FDE *= config.WORLD_SCALE
            ade = ADE.mean()
            fde = FDE.mean()
            sys.stdout.write("\r\033[K ADE: {:.4f}; FDE: {:.4f} -- time: {}s".format(
                ade, fde,
                int(time.time()-tic))
            )
            print()
            return ade, fde


    ###############################################################################
    #####                                                                    ######
    ##### load model                                                         ######
    #####                                                                    ######
    ###############################################################################
    model = MetaPTP(horizon=config.PRED_HORIZON, ob_radius=config.OB_RADIUS, hidden_dim=config.RNN_HIDDEN_DIM)
    model.to(settings.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    start_epoch = 0
    if settings.ckpt:
        ckpt = os.path.join(settings.ckpt, "ckpt-last")
        ckpt_best = os.path.join(settings.ckpt, "ckpt-best")
        if os.path.exists(ckpt_best):
            state_dict = torch.load(ckpt_best, map_location=settings.device)
            ade_best = state_dict["ade"]
            fde_best = state_dict["fde"]
        else:
            ade_best = 100000
            fde_best = 100000
            fpc_best = 1
        if train_data is None: # testing mode
            ckpt = ckpt_best
        if os.path.exists(ckpt):
            print("Load from ckpt:", ckpt)
            state_dict = torch.load(ckpt, map_location=settings.device)
            model.load_state_dict(state_dict["model"])
            if "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
                rng_state = [r.to("cpu") if torch.is_tensor(r) else r for r in state_dict["rng_state"]]
            start_epoch = state_dict["epoch"]
    end_epoch = start_epoch+1 if train_data is None or start_epoch >= config.EPOCHS else config.EPOCHS

    if settings.train and settings.ckpt:
        logger = SummaryWriter(log_dir=settings.ckpt)
    else:
        logger = None



    for epoch in range(start_epoch+1, end_epoch+1):

        losses = None

        ade, fde = 10000, 10000
        perform_test = (train_data is None or epoch >= config.TEST_SINCE) and test_data is not None
        if perform_test:
            ade, fde = test(model)


        if losses is not None and settings.ckpt:
            if logger is not None:
                for k, v in losses.items():
                    logger.add_scalar("train/{}".format(k), v, epoch)
                if perform_test:
                    logger.add_scalar("eval/ADE", ade, epoch)
                    logger.add_scalar("eval/FDE", fde, epoch)
            state = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                ade=ade, fde=fde, epoch=epoch, rng_state=rng_state
            )
            torch.save(state, ckpt)
            if ade < ade_best:
                ade_best = ade
                fde_best = fde
                best_ep = epoch
                state = dict(
                    model=state["model"],
                    ade=ade, fde=fde, epoch=epoch
                )
                torch.save(state, ckpt_best)
    
