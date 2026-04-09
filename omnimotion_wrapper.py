import os
import torch
import torch.nn.functional as F
import numpy as np
import imageio
import argparse
import json
from typing import List, Union, Dict
from tqdm.auto import tqdm
from os.path import join, dirname
from torch.utils.tensorboard import SummaryWriter

from .trainer import BaseTrainer
from .loaders.create_training_dataset import get_training_dataset

# Preprocessing related imports

from .preprocessing.RAFT.core.raft import RAFT
from .preprocessing.RAFT.core.utils.utils import InputPadder
from .preprocessing.dino.utils import load_pretrained_weights
from .preprocessing.dino import vision_transformer as vits

def gen_grid(h, w, device):
    lin_y = torch.arange(0, h, device=device)
    lin_x = torch.arange(0, w, device=device)
    grid_y, grid_x = torch.meshgrid((lin_y, lin_x), indexing='ij')
    grid = torch.stack((grid_x, grid_y), -1)
    return grid  # [h, w, 2]

def normalize_coords(coords, h, w):
    return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2 - 1.

class OmniMotionOptimizer:
    def __init__(
        self, 
        video_tensor: torch.Tensor, 
        mean: List[float] = [0.485, 0.456, 0.406], 
        std: List[float] = [0.229, 0.224, 0.225],
        device: str = 'cuda',
        expname: str = 'omnimotion_mem_exp'
    ):
        self.device = device
        self.expname = expname
        
        # [N, 3, H, W] -> [N, H, W, 3] in [0, 1]
        mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        std_t = torch.tensor(std).view(1, 3, 1, 1)
        images_01 = (video_tensor.cpu() * std_t + mean_t).clamp(0, 1)
        self.images = images_01.permute(0, 2, 3, 1) # [N, H, W, 3]
        self.images_gpu = self.images.to(device)
        self.num_imgs, self.h, self.w, _ = self.images.shape
        
        self.flows = {}
        self.raft_masks = {}
        self.sample_weights = {}
        self.features = [] # DINO features
        
        self.args = self._setup_args()
        self.trainer = None

    def _setup_args(self):
        args = argparse.Namespace()
        args.data_dir = 'memory' 
        args.expname = self.expname
        args.save_dir = './out'
        args.ckpt_path = ''
        args.no_reload = False
        args.distributed = 0
        args.local_rank = 0
        args.num_iters = 100000
        args.num_workers = 0 
        args.load_opt = 1
        args.load_scheduler = 1
        args.loader_seed = 12
        args.dataset_types = 'flow'
        args.dataset_weights = [1.0]
        args.num_imgs = self.num_imgs
        args.num_pairs = 8
        args.num_pts = 256
        args.lr_feature, args.lr_deform, args.lr_color = 1e-3, 1e-4, 3e-4
        args.lrate_decay_steps, args.lrate_decay_factor = 20000, 0.5
        args.grad_clip = 0
        args.use_error_map = args.use_count_map = args.use_affine = args.mask_near = False
        args.num_samples_ray, args.pe_freq = 32, 4
        args.min_depth, args.max_depth = 0, 2
        args.start_interval, args.max_padding = 20, 0
        args.chunk_size, args.use_max_loc = 10000, True
        args.query_frame_id, args.vis_occlusion = 0, False
        args.occlusion_th, args.foreground_mask_path = 0.99, ''
        args.i_print, args.i_img, args.i_weight, args.i_cache = 100, 500, 20000, 20000
        
        args.images = self.images
        args.flows = self.flows
        args.raft_masks = self.raft_masks
        args.sample_weights = self.sample_weights
        return args

    def preprocess_full_in_memory(
        self, 
        save_dir: str,
        video_name: str,
        raft_model_path=join(dirname(__file__), 'preprocessing/RAFT/models/raft-things.pth'),
        # dino_model_path=join(dirname(__file__), 'preprocessing/dino/dino_vit_small_patch16_teacher.pth'),
        cycle_th=3.0,
        run_chaining=True
    ):
        os.makedirs(save_dir, exist_ok=True)
        
        # 4. Chaining (Optional)
        chain_cache = os.path.join(save_dir, f"chained_data_{video_name}.npz")
        weights_cache = os.path.join(save_dir, f"sample_weights_{video_name}.json")
        if run_chaining and os.path.exists(chain_cache):
            print(f"Loading cached chained data from {chain_cache}")
            loaded_chain = np.load(chain_cache)
            # Chaining overwrites flows and masks
            flow_keys = [k for k in loaded_chain.files if not k.endswith('_mask')]
            self.flows.update({k: loaded_chain[k] for k in flow_keys})
            self.raft_masks.update({k.replace('_mask', ''): loaded_chain[k] for k in loaded_chain.files if k.endswith('_mask')})
            with open(weights_cache, 'r') as f:
                self.sample_weights.update(json.load(f))
        else:
            # 1. RAFT Flow
            raft_cache = os.path.join(save_dir, f"raft_exhaustive_{video_name}.npz")
            if os.path.exists(raft_cache):
                print(f"Loading cached RAFT flows from {raft_cache}")
                loaded = np.load(raft_cache)
                self.flows.update({k: loaded[k] for k in loaded.files})
            else:
                self._compute_exhaustive_raft(raft_model_path)
                np.savez_compressed(raft_cache, **self.flows)

        # 2. DINO Features
        dino_cache = os.path.join(save_dir, f"dino_features_{video_name}.pt")
        if os.path.exists(dino_cache):
            print(f"Loading cached DINO features from {dino_cache}")
            self.features = torch.load(dino_cache)
        else:
            self._extract_dino_features()
            torch.save(self.features, dino_cache)

        if run_chaining and os.path.exists(chain_cache):
            pass
        else:
            # 3. Filtering
            mask_cache = os.path.join(save_dir, f"raft_masks_{video_name}.npz")
            if os.path.exists(mask_cache) and os.path.exists(weights_cache):
                print(f"Loading cached masks and weights from {mask_cache}")
                loaded_masks = np.load(mask_cache)
                self.raft_masks.update({k: loaded_masks[k] for k in loaded_masks.files})
                with open(weights_cache, 'r') as f:
                    self.sample_weights.update(json.load(f))
            else:
                self._filter_raft(cycle_th)
                np.savez_compressed(mask_cache, **self.raft_masks)
                with open(weights_cache, 'w') as f:
                    json.dump(self.sample_weights, f)

        # 4. Chaining (Optional)
        if run_chaining:
            chain_cache = os.path.join(save_dir, f"chained_data_{video_name}.npz")
            if os.path.exists(chain_cache):
                pass
            else:
                self._chain_raft()
                # Save both flows and masks since they were modified
                chain_save_dict = {**self.flows}
                for k, v in self.raft_masks.items():
                    chain_save_dict[f"{k}_mask"] = v
                np.savez_compressed(chain_cache, **chain_save_dict)
                # Re-save weights
                with open(weights_cache, 'w') as f:
                    json.dump(self.sample_weights, f)

    def _compute_exhaustive_raft(self, model_path):
        if RAFT is None: raise ImportError("RAFT module not found")
        print("Computing Exhaustive RAFT Flow...")
        raft_args = argparse.Namespace(model=model_path, small=False, mixed_precision=False, alternate_corr=False)
        model = torch.nn.DataParallel(RAFT(raft_args))
        model.load_state_dict(torch.load(model_path))
        model = model.module.to(self.device).eval()

        img_names = [f'{i:05d}.jpg' for i in range(self.num_imgs)]
        with torch.no_grad():
            for i in tqdm(range(self.num_imgs), desc="Exhaustive RAFT"):
                for j in range(self.num_imgs):
                    if i == j: continue
                    image1 = self.images_gpu[i].permute(2, 0, 1)[None] * 255
                    image2 = self.images_gpu[j].permute(2, 0, 1)[None] * 255
                    padder = InputPadder(image1.shape)
                    img1, img2 = padder.pad(image1, image2)
                    _, flow_up = model(img1, img2, iters=20, test_mode=True)
                    flow_up = padder.unpad(flow_up).squeeze().permute(1, 2, 0).cpu().numpy()
                    self.flows[f'{img_names[i]}_{img_names[j]}'] = flow_up

    def _extract_dino_features(self):
        if vits is None: raise ImportError("DINO (vits) module not found")
        print("Extracting DINO features...")
        model = vits.__dict__['vit_small'](patch_size=16, num_classes=0)
        # state_dict = torch.load(model_path)
        # if 'teacher' in state_dict: state_dict = state_dict['teacher']
        # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # model.load_state_dict(state_dict, strict=False)
        # model.to(self.device).eval()

        model.to(self.device)
        load_pretrained_weights(model, dirname(__file__), "teacher", "vit_small", 16)
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        for i in tqdm(range(self.num_imgs), desc="DINO Features"):
            frame = self.images_gpu[i].permute(2, 0, 1)[None] # [1, 3, H, W]
            # Normalize for DINO
            norm_frame = F.interpolate(frame, size=((self.h // 16) * 16, (self.w // 16) * 16), mode='bilinear')
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.228, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            norm_frame = (norm_frame - mean) / std
            
            with torch.no_grad():
                out = model.get_intermediate_layers(norm_frame, n=1)[0]
                out = out[:, 1:, :] 
                feat_h, feat_w = norm_frame.shape[2] // 16, norm_frame.shape[3] // 16
                out = out[0].reshape(feat_h, feat_w, -1)
            self.features.append(out) # [feat_h, feat_w, dim]

    def _filter_raft(self, cycle_th):
        print("Filtering RAFT flows...")
        grid = gen_grid(self.h, self.w, self.device).permute(2, 0, 1)[None].float() # [1, 2, H, W]
        img_names = [f'{i:05d}.jpg' for i in range(self.num_imgs)]
        
        for i in tqdm(range(self.num_imgs), desc="Filtering"):
            self.sample_weights[img_names[i]] = {}
            feat_i = self.features[i].permute(2, 0, 1)[None]
            feat_i_sampled = F.interpolate(feat_i, size=(self.h, self.w), mode='bilinear').permute(0, 2, 3, 1)[0]

            for j in range(self.num_imgs):
                if i == j: continue
                key_fg = f'{img_names[i]}_{img_names[j]}'
                key_bg = f'{img_names[j]}_{img_names[i]}'
                flow_f = torch.from_numpy(self.flows[key_fg]).to(self.device).permute(2, 0, 1)[None]
                flow_b = torch.from_numpy(self.flows[key_bg]).to(self.device).permute(2, 0, 1)[None]

                coord2 = grid + flow_f
                coord2_normed = normalize_coords(coord2.permute(0, 2, 3, 1), self.h, self.w)
                flow_21_sampled = F.grid_sample(flow_b, coord2_normed, align_corners=True)
                fb_discrepancy = torch.norm((flow_f + flow_21_sampled).squeeze(), dim=0)
                mask_cycle = fb_discrepancy < cycle_th

                feat_j = self.features[j].permute(2, 0, 1)[None]
                feat_j_sampled = F.grid_sample(feat_j, coord2_normed, align_corners=True).permute(0, 2, 3, 1)[0]
                feature_sim = torch.cosine_similarity(feat_i_sampled, feat_j_sampled, dim=-1)
                feature_mask = feature_sim > 0.5
                
                if abs(i - j) >= 3: mask_cycle = mask_cycle * feature_mask

                if abs(i - j) < 3:
                    map_i = flow_f + flow_21_sampled
                    coord_21 = grid + map_i  # [1, 2, h, w]
                    coord_21_normed = normalize_coords(coord_21.squeeze().permute(1, 2, 0), self.h, self.w)  # [h, w, 2]
                    flow_22 = F.grid_sample(flow_f, coord_21_normed[None], align_corners=True)
                    fbf_discrepancy = torch.norm((coord_21 + flow_22 - flow_f - grid).squeeze(), dim=0)
                    mask_in_range = (coord2_normed.min(dim=-1)[0] >= -1) * (coord2_normed.max(dim=-1)[0] <= 1)
                    mask_occluded = (fbf_discrepancy < cycle_th) * (fb_discrepancy > cycle_th * 1.5)
                    mask_occluded *= mask_in_range.squeeze()
                else:
                    mask_occluded = torch.zeros_like(mask_cycle)

                mask = torch.stack([mask_cycle, mask_occluded, torch.zeros_like(mask_cycle)], dim=-1).cpu().numpy()
                self.raft_masks[key_fg] = mask
                self.sample_weights[img_names[i]][img_names[j]] = np.sum(mask).item()

    def _chain_raft(self):
        print("Chaining RAFT flows...")
        img_names = [f'{i:05d}.jpg' for i in range(self.num_imgs)]
        grid = gen_grid(self.h, self.w, self.device)[None].float()
        
        def process_chain(indices):
            for i in tqdm(indices[:-1], desc="Chaining"):
                key_start = f'{img_names[i]}_{img_names[indices[indices.index(i)+1]]}'
                accum_flow = torch.from_numpy(self.flows[key_start]).to(self.device)[None]
                accum_mask = self.raft_masks[key_start][..., 0] > 0
                feat_i = F.interpolate(self.features[i].permute(2, 0, 1)[None], size=(self.h, self.w), mode='bilinear')

                for j in indices[indices.index(i)+1:]:
                    key_curr = f'{img_names[i]}_{img_names[j]}'
                    direct_masks = self.raft_masks[key_curr]
                    direct_cycle, direct_occ = direct_masks[..., 0] > 0, direct_masks[..., 1] > 0
                    
                    # Update with direct if available
                    direct_mask_t = torch.from_numpy(direct_cycle | direct_occ).to(self.device)
                    direct_flow_t = torch.from_numpy(self.flows[key_curr]).to(self.device)
                    accum_flow[0][direct_mask_t] = direct_flow_t[direct_mask_t]

                    coords_normed = normalize_coords(grid + accum_flow, self.h, self.w)
                    feat_j_samp = F.grid_sample(self.features[j].permute(2, 0, 1)[None], coords_normed, align_corners=True)
                    feat_sim = torch.cosine_similarity(feat_i, feat_j_samp, dim=1).squeeze(0).cpu().numpy()
                    img_j_samp = F.grid_sample(self.images_gpu[j].permute(2, 0, 1)[None], coords_normed, align_corners=True).squeeze()
                    rgb_sim = torch.norm(self.images_gpu[i].permute(2, 0, 1) - img_j_samp, dim=0).cpu().numpy()

                    accum_mask = accum_mask * (feat_sim > 0.5) * (rgb_sim < 0.3)
                    accum_mask[direct_cycle] = True
                    accum_mask[direct_occ] = False

                    self.flows[key_curr] = accum_flow[0].cpu().numpy()
                    new_mask = np.stack([accum_mask, direct_occ, np.zeros_like(accum_mask)], axis=-1)
                    self.raft_masks[key_curr] = new_mask
                    self.sample_weights[img_names[i]][img_names[j]] = np.sum(new_mask).item()

                    if (indices == range(self.num_imgs) and j < self.num_imgs - 1) or \
                       (indices == range(self.num_imgs-1, -1, -1) and j > 0):
                        next_j = indices[indices.index(j) + (1 if indices == range(self.num_imgs) else -1)]
                        flow_next = torch.from_numpy(self.flows[f'{img_names[j]}_{img_names[next_j]}']).to(self.device).permute(2, 0, 1)[None]
                        accum_flow += F.grid_sample(flow_next, coords_normed, align_corners=True).permute(0, 2, 3, 1)
                        m_next = self.raft_masks[f'{img_names[j]}_{img_names[next_j]}'][..., 0] > 0
                        m_next_samp = F.grid_sample(torch.from_numpy(m_next).float()[None, None].to(self.device), coords_normed, align_corners=True).squeeze().cpu().numpy() == 1
                        accum_mask *= m_next_samp

        process_chain(list(range(self.num_imgs)))
        process_chain(list(range(self.num_imgs-1, -1, -1)))

    def run_optimization(self, num_iters: int = 1000, output_dir = "."):
        self.args.num_iters = num_iters
        writer = SummaryWriter(output_dir)
        dataset, data_sampler = get_training_dataset(self.args, max_interval=self.args.start_interval)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.num_pairs, num_workers=0)
        if self.trainer is None: self.trainer = BaseTrainer(self.args, device=self.device)
        start_step = self.trainer.step + 1
        for step in tqdm(range(start_step, start_step + num_iters)):
            batch = next(iter(data_loader))
            self.trainer.train_one_step(step, batch)
            if step % self.args.i_print == 0:
                print(f"Step {step}: loss = {self.trainer.scalars_to_log.get('loss/Loss', 'N/A')}")
                self.save_checkpoint(f"{output_dir}/{step}.pt")
            self.trainer.log(writer, step)
            dataset.set_max_interval(self.args.start_interval + step // 2000)
        writer.flush()
        writer.close()

    def query_trajectory(self, query_frame_id: int, points: Union[torch.Tensor, np.ndarray]):
        if self.trainer is None: self.trainer = BaseTrainer(self.args, device=self.device)
        if isinstance(points, np.ndarray): points = torch.from_numpy(points).float().to(self.device)
        trajs = []
        depths = []
        with torch.no_grad():
            for tid in range(self.num_imgs):
                # if tid == query_frame_id: trajs.append(points)
                # else:
                p2, d2 = self.trainer.get_correspondences_for_pixels(ids1=[query_frame_id], px1s=points[None], ids2=[tid], use_max_loc=True, return_depth=True)
                trajs.append(p2[0])
                depths.append(d2[0])
        return torch.stack(trajs, dim=0), torch.stack(depths, dim=0)

    def save_checkpoint(self, path: str): self.trainer.save_model(path)
    def load_checkpoint(self, path: str):
        if self.trainer is None: self.trainer = BaseTrainer(self.args, device=self.device)
        self.trainer.load_model(path)
