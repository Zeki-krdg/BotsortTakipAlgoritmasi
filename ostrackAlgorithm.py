import cv2
import numpy as np
import torch
import math
import torch.nn.functional as F
from ostrack_model import OSTrack

class OptimizedOSTrack:
    """    
    Real OSTrack (One-Stream Tracker) Interface
    Supports both ViT-Base and ViT-Small.
    """       
    def __init__(self, model_type='base', weight_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[OSTrack] Initializing Transformer Tracker (ViT-{model_type}) on {self.device}...")
        
        # Load Model
        self.model = OSTrack(model_type=model_type)
        if weight_path:
            try:
                state_dict = torch.load(weight_path, map_location='cpu')
                # Strict=False to bypass mismatches if loading official weights directly into this simplified model
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[OSTrack] Loaded weights from {weight_path}")
            except Exception as e:
                print(f"[OSTrack] Warning: Failed to load weights ({e}). Running without pretrained weights!")
        else:
            print("[OSTrack] Warning: No weight_path provided. Model will have random weights. This will NOT track well!")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Tracking states
        self.template_tensor = None
        self.state = None # [cx, cy, w, h] of current target
        self.template_size = 128
        self.search_size = 256
        self.search_area_factor = 4.0 # Search region size proportional to target size
        self.frame_id = 0
        self.score_ema = None
        self.lost_streak = 0

        # Image norm constants used in PyTorch pretraining (ImageNet standard)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def _clamp_bbox(self, bbox, img_w, img_h):
        x, y, w, h = bbox
        x = max(0, min(int(x), img_w - 1))
        y = max(0, min(int(y), img_h - 1))
        w = max(2, min(int(w), img_w - x))
        h = max(2, min(int(h), img_h - y))
        return (x, y, w, h)

    def _update_template(self, frame):
        """Update template slowly only when tracking is highly confident."""
        cx, cy, _, _ = self.state
        current_template = self._preprocess_crop(frame, self.template_size, cx, cy, self.sz)
        alpha = 0.03
        self.template_tensor = (1.0 - alpha) * self.template_tensor + alpha * current_template

    def _preprocess_crop(self, frame, crop_size, cx, cy, sz):
        """ Crop an image centered at cx, cy area mapping to sz*sz, then resizing to crop_size """
        img_h, img_w = frame.shape[:2]
        pad = int(sz // 2)
        
        # Get coordinates within image
        x1 = int(cx - pad)
        y1 = int(cy - pad)
        x2 = int(cx + pad)
        y2 = int(cy + pad)
        
        # Handle borders via padding if out of bounds
        req_pad_l = max(0, -x1)
        req_pad_t = max(0, -y1)
        req_pad_r = max(0, x2 - img_w)
        req_pad_b = max(0, y2 - img_h)
        
        if req_pad_l > 0 or req_pad_t > 0 or req_pad_r > 0 or req_pad_b > 0:
            frame_padded = cv2.copyMakeBorder(frame, req_pad_t, req_pad_b, req_pad_l, req_pad_r, cv2.BORDER_CONSTANT, value=(127,127,127))
            x1 += req_pad_l
            y1 += req_pad_t
            x2 += req_pad_l
            y2 += req_pad_t
        else:
            frame_padded = frame
            
        crop = frame_padded[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (crop_size, crop_size))
        
        # BGR to RGB
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        
        # To tensor
        tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Normalize
        tensor = (tensor - self.mean) / self.std
        return tensor

    def init(self, frame, bbox):
        """ 
        Initialize tracker with template image 
        bbox format: (x, y, w, h)
        """
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        
        # Size in original image for cropping template
        sz = math.sqrt(w * h * self.search_area_factor)
        
        self.state = [cx, cy, w, h]
        self.sz = sz
        self.frame_id = 0
        self.score_ema = None
        self.lost_streak = 0
        
        # Crop template
        self.template_tensor = self._preprocess_crop(frame, self.template_size, cx, cy, sz)
        return True

    @torch.no_grad()
    def track(self, frame, full_frame_search=False):
        """ 
        Track target in the new frame 
        full_frame_search is ignored in pure OSTrack as Transformer relies on local patches,
        but we implement a search scale multiplier if full frame is requested.
        """
        if self.template_tensor is None or self.state is None:
            return None, 0
            
        self.frame_id += 1
        cx, cy, w, h = self.state
        
        sz_search = self.sz
        if full_frame_search:
            sz_search *= 1.8 + min(0.6, 0.1 * self.lost_streak)
        
        # Crop search region
        search_tensor = self._preprocess_crop(frame, self.search_size, cx, cy, sz_search)
        
        # Forward pass
        score_map, offset_map, size_map = self.model(self.template_tensor, search_tensor)
        
        # Process output maps
        # Score map: [1, 1, 16, 16]
        score_map = score_map[0, 0] # [16, 16]
        offset_map = offset_map[0] # [2, 16, 16]
        size_map = size_map[0] # [2, 16, 16]
        
        # Find peak
        max_idx = torch.argmax(score_map)
        py = (max_idx // score_map.shape[1]).item()
        px = (max_idx % score_map.shape[1]).item()
        
        score_raw = score_map[py, px].item()
        
        # Get offset and size at peak
        offset_x = offset_map[0, py, px].item()
        offset_y = offset_map[1, py, px].item()
        
        pred_w = size_map[0, py, px].item()
        pred_h = size_map[1, py, px].item()
        
        # Map back to cropped image coordinate (0 to 256)
        patch_size = self.search_size / score_map.shape[0]  # usually 256/16 = 16
        crop_cx = (px + offset_x) * patch_size
        crop_cy = (py + offset_y) * patch_size
        
        # Scale back to original frame coordinate
        scale = sz_search / self.search_size
        
        cx = cx + (crop_cx - self.search_size / 2) * scale
        cy = cy + (crop_cy - self.search_size / 2) * scale
        new_w = pred_w * self.search_size * scale
        new_h = pred_h * self.search_size * scale
        
        # Motion consistency: large center jumps are usually drift.
        prev_diag = math.sqrt(max(4.0, w * h))
        jump = math.sqrt((cx - self.state[0]) ** 2 + (cy - self.state[1]) ** 2)
        jump_ratio = jump / (prev_diag + 1e-6)
        motion_conf = max(0.0, 1.0 - min(1.0, jump_ratio / 2.0))

        # Final confidence combines model score and motion consistency.
        score = 0.7 * score_raw + 0.3 * motion_conf
        if self.score_ema is None:
            self.score_ema = score
        else:
            self.score_ema = 0.85 * self.score_ema + 0.15 * score
        conf = 0.6 * score + 0.4 * self.score_ema

        if conf < 0.45:
            self.lost_streak += 1
        else:
            self.lost_streak = 0

        # Update state smoothly (lower lr helps prevent rapid size explosion).
        lr = max(0.15, min(0.75, 0.2 + conf * 0.5))
        self.state[0] = cx
        self.state[1] = cy
        self.state[2] = self.state[2] * (1 - lr) + new_w * lr
        self.state[3] = self.state[3] * (1 - lr) + new_h * lr
        self.sz = math.sqrt(max(4.0, self.state[2] * self.state[3] * self.search_area_factor))
        
        # Return format (x, y, w, h)
        out_x = cx - self.state[2] / 2
        out_y = cy - self.state[3] / 2
        
        img_h, img_w = frame.shape[:2]
        bbox = self._clamp_bbox((out_x, out_y, self.state[2], self.state[3]), img_w, img_h)

        # Update template only when confidence is high.
        if conf > 0.75 and (self.frame_id % 3 == 0):
            self._update_template(frame)
        
        return bbox, conf