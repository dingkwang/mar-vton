import os
import numpy as np

import torch
import torchvision.datasets as datasets

import os
from typing import List, Union, Dict
import shutil
from copy import deepcopy

import json
import PIL.Image as Image
import cv2
import numpy as np
import math
import random

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms
import torch.nn.functional as F


# --------------------------------------------- #
#             Preprocess Data Utils
# --------------------------------------------- #
def make_dir(dir):
    """remove the dir if it exists, and create a new one"""
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def check_datafolder(target_dir: str):
    all_img_names = os.listdir(target_dir)
    all_img_names = set([x.split('_')[0] for x in all_img_names])
    print(f"In {target_dir}, there are {len(all_img_names)} distinct images", "\n")
    # print(all_img_names)

def img2tensor(img):
    transform_validation = transforms.Compose([
        transforms.ToTensor()])
    return transform_validation(img)

def mv_image(src_dir, img_list, trg_dir):
    """Mave the images in the list from source dir to target dir.
    All images will be resized to (768, 1024) if not.
    """
    # Recreat the target dir.
    make_dir(trg_dir)
    for file in img_list:
        img = Image.open(f"{src_dir}/{file}")
        if img.size != (768, 1024):
            img = img.resize((768, 1024))
            img.save(f"{trg_dir}/{file}")
        else:
            shutil.copy(os.path.join(src_dir, file), os.path.join(trg_dir, file))
        
    # check the target dir
    check_datafolder(trg_dir)

def generate_bbox(src_folder, img_list, trg_folder, save: bool = False):
    """Load the mask img and Generate the bounding box.
    """
    if save:
        # Recreat the target dir.
        make_dir(trg_folder)
    bbox_dict = {}
    for img_name in img_list:
        image_path = os.path.join(src_folder, img_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        # Load the image (assuming it's a grayscale image)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Threshold the image to get the mask (assuming white is 255)
        _, mask = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)

        # Find contours of the masked area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the bounding rectangle of the largest contour
        bounding_box = [0, 0, 0, 0]
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            bounding_box = [x, y, x + w, y + h]

            # Draw a gray rectangle edge over the masked area
            gray_color = 128  # Gray color in grayscale
            cv2.rectangle(image, (x, y), (x + w, y + h), gray_color, 2)  # 2 is the thickness of the edge
        else:
            bounding_box = "No contour found"
        bbox_dict[img_name] = bounding_box
        # Save the result
        if save:
            result_image_path = os.path.join(trg_folder, img_name)
            cv2.imwrite(result_image_path, image)

            json_path = os.path.join(trg_folder, 'bbox_dict.json')
            with open(json_path, 'w') as json_file:
                json.dump(bbox_dict, json_file, indent=4)

    check_datafolder(trg_folder)

def apply_mask(model_folder, img_list, mask_folder, trg_folder, save: bool = False):
    """Apply the mask to the model image."""
    # Recreat the target dir.
    make_dir(trg_folder)
    for img_name in img_list:
        model = Image.open(f"{model_folder}/{img_name}")
        mask = Image.open(f"{mask_folder}/{img_name}")
        mask = mask.convert('RGB')

        # Convert images to PyTorch tensors
        model_tensor = img2tensor(model)
        mask_tensor = img2tensor(mask)

        # Gray out the masked area in the masked image tensor
        mask_tensor = torch.round(mask_tensor).to(torch.int)
        masking = (mask_tensor > 0)
        masked_image_tensor = torch.ones_like(model_tensor) * 0.5
        masked_image_tensor = torch.where(masking, masked_image_tensor, model_tensor)

        # Convert the masked image tensor back to a PIL image
        # masked_image = Image.fromarray(masked_image_tensor.numpy().astype(np.uint8).transpose(1, 2, 0))
        transform = transforms.ToPILImage()
        masked_image = transform(masked_image_tensor)

        # Save the masked image
        os.makedirs(trg_folder, exist_ok=True)
        result_image_path = f"{trg_folder}/{img_name}"
        masked_image.save(result_image_path)
    
def apply_warped_cloth(model_folder, img_list, mask_folder, warped_folder, trg_folder, save: bool = False):
    """Apply the warped clothes to agnostic model images."""
    # Recreate the target dir.
    make_dir(trg_folder)
    for img_name in img_list:
        model = Image.open(f"{model_folder}/{img_name}")
        cloth = Image.open(f"{warped_folder}/{img_name}")
        mask = Image.open(f"{mask_folder}/{img_name}")
        mask = mask.convert('L')  # Convert mask to grayscale

        # Convert images to PyTorch tensors
        model_tensor = img2tensor(model)
        cloth_tensor = img2tensor(cloth)
        mask_tensor = img2tensor(mask)

        # Ensure all tensors have the same dimensions
        if model_tensor.shape[1:] != cloth_tensor.shape[1:] or model_tensor.shape[1:] != mask_tensor.shape[1:]:
            raise ValueError("Model, cloth, and mask dimensions do not match")

        # Apply the mask
        result = torch.where(mask_tensor > 0.5, cloth_tensor, model_tensor)

        # Convert the result tensor back to a PIL image
        transform = transforms.ToPILImage()
        result_image = transform(result)

        # Save the result
        if save:
            result_image_path = os.path.join(trg_folder, img_name)
            result_image.save(result_image_path)

    check_datafolder(trg_folder)


    check_datafolder(trg_folder)

# --------------------------------------------- #
#                  Data Set Utils
# --------------------------------------------- #

# Function to split dataset
def split_dataset(dataset, train_ratio=0.8):
    if train_ratio == 1.0:
        return dataset, dataset
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_names = [img for img in os.listdir(data_dir) if img.endswith('.jpg')]  # Get a list of image file names

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.img_names[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, "handler_for_labels"

class CpDataset(torch.utils.data.Dataset):
    """The consistency between model, garment, and masked model images is guaranteed by the model image's name."""
    def __init__(self, data_dir, transform=None, use_bbox=True):
        self.data_dir = data_dir
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor()])
        # cache all image names
        self.img_names = [img for img in os.listdir(os.path.join(data_dir, "model")) if img.endswith('.jpg')]
        self.use_bbox = use_bbox
        # cache bbox coordinates
        if use_bbox:
            bbox_dir = os.path.join(data_dir, "mask_bbox")
            with open(os.path.join(bbox_dir, "bbox_dict.json"), "r") as f:
                self.bbox_dict = json.load(f)

        # resize bbox if the img will be resized.
        size = (768, 1024)
        # get the size used in transforms
        for t in self.transform.transforms:
            if isinstance(t, transforms.transforms.Resize):
                size = t.size
                break
        if use_bbox:
            bbox_unfound_img_names = []
            for i, img_name in enumerate(self.img_names):
                result = self.bbox_dict[img_name]
                if result == "No contour found":
                    bbox_unfound_img_names.append((i, img_name))
                    continue
                if size == (768, 1024):
                    return torch.tensor(result)
                else:
                    result[0] = result[0] * size[0] / 768
                    result[1] = result[1] * size[1] / 1024
                    result[2] = result[2] * size[0] / 768
                    result[3] = result[3] * size[1] / 1024

        # remove the unfound bbox images
        if use_bbox:
            for i, img_name in bbox_unfound_img_names:
                self.img_names.pop(i)
                self.bbox_dict.pop(img_name)

    def __len__(self):
        return len(self.img_names)
    
    def __get_img__(self, idx, subfolder):
        img_name = os.path.join(self.data_dir, subfolder, self.img_names[idx])
        image = Image.open(img_name).convert('RGB')
        return self.transform(image)
    
    def __get_bbox__(self, idx):
        img_name = self.img_names[idx]
        result = self.bbox_dict[img_name]
        return torch.tensor(result, dtype=torch.int32)

    # TODO: include bbox here
    def __getitem__(self, idx):
        gt = self.__get_img__(idx, "model")
        garment = self.__get_img__(idx, "garment")
        inpaint = self.__get_img__(idx, "inpaint")
        # inpaint_warped = self.__get_img__(idx, "inpaint_warped_cloth")
        if self.use_bbox:
            bbox = self.__get_bbox__(idx)
        # All values in the result dict should be tensors
        result = {
            'gt': gt, 
            'garment': garment,
            'inpaint': inpaint,
            # 'inpaint_warped': inpaint_warped,
            'bbox': bbox
        }
        return result, "handler_for_labels"

# --------------------------------------------- #
#                  Inference Utils
# --------------------------------------------- #
def mask_by_random_topk(mask_len, probs, temperature=1.0):
    mask_len = mask_len.squeeze()
    confidence = torch.log(probs) + temperature * torch.Tensor(np.random.gumbel(size=probs.shape)).cuda()
    sorted_confidence, _ = torch.sort(confidence, dim=-1)
    
    # Obtain cut-off thresholds for all samples in the batch
    bsz = probs.shape[0]
    mask_len_long = mask_len.long()
    row_indices = torch.arange(bsz).cuda()
    cut_off = sorted_confidence[row_indices, mask_len_long - 1].unsqueeze(1)
    
    # Masks tokens with lower confidence
    masking = (confidence <= cut_off)
    
    # Check if all tokens are masked for any sample in the batch
    if masking.all(dim=1).any():
        raise ValueError("All tokens are masked for at least one sample in the batch!")
    
    return masking

def get_sample_stats(sampled_ids, cur_ids, gt_indices, logits, mask_token_id, step):
    """Get the loss and accuracy of the sampled tokens at the given step."""
    bsz, seq_len = gt_indices.size()
    
    # Loss and accuracy of all samples
    sampled_loss = F.cross_entropy(logits.reshape(bsz*seq_len, -1), gt_indices.reshape(bsz*seq_len))
    sampled_accuracy = (sampled_ids == gt_indices).float().mean()
    
    # Create masks for masked and kept tokens
    mask = (cur_ids == mask_token_id)
    kept_mask = ~mask
    
    # Compute masked stats
    masked_logits = logits[mask].reshape(-1, logits.size(-1))
    masked_gt = gt_indices[mask]
    masked_sampled = sampled_ids[mask]
    
    masked_loss = F.cross_entropy(masked_logits, masked_gt)
    masked_accuracy = (masked_sampled == masked_gt).float().mean()
    
    _, top3_indices = masked_logits.topk(3, dim=-1)
    _, top5_indices = masked_logits.topk(5, dim=-1)
    
    masked_top3_accuracy = (top3_indices == masked_gt.unsqueeze(-1)).float().sum(dim=-1).mean()
    masked_top5_accuracy = (top5_indices == masked_gt.unsqueeze(-1)).float().sum(dim=-1).mean()
    
    # Compute kept stats
    kept_logits = logits[kept_mask].reshape(-1, logits.size(-1))
    kept_gt = gt_indices[kept_mask]
    kept_sampled = sampled_ids[kept_mask]
    
    kept_loss = F.cross_entropy(kept_logits, kept_gt)
    kept_accuracy = (kept_sampled == kept_gt).float().mean()
    
    _, kept_top3_indices = kept_logits.topk(3, dim=-1)
    _, kept_top5_indices = kept_logits.topk(5, dim=-1)
    
    kept_top3_accuracy = (kept_top3_indices == kept_gt.unsqueeze(-1)).float().sum(dim=-1).mean()
    kept_top5_accuracy = (kept_top5_indices == kept_gt.unsqueeze(-1)).float().sum(dim=-1).mean()
    
    return {
        'step': step, 
        'loss': sampled_loss.item(), 
        'masked_loss': masked_loss.item(), 
        'kept_loss': kept_loss.item(), 
        'accuracy': sampled_accuracy.item(), 
        'masked_accuracy': masked_accuracy.item(), 
        'masked_top3_accuracy': masked_top3_accuracy.item(),
        'masked_top5_accuracy': masked_top5_accuracy.item(),
        'kept_accuracy': kept_accuracy.item(),
        'kept_top3_accuracy': kept_top3_accuracy.item(),
        'kept_top5_accuracy': kept_top5_accuracy.item(),
        'masked_len': mask.sum().item(), 
        'kept_len': kept_mask.sum().item()
    }

def get_encoded_results(model, img):
    with torch.no_grad():
        z_q, _, token_tuple = model.vqgan.encode(img)

    _, _, token_indices = token_tuple
    token_indices = token_indices.reshape(z_q.size(0), -1)
    token_indices = token_indices.clone().detach().long()

    return z_q, token_indices

def sample_tryon(model, batch: dict, args, num_iter=12, choice_temperature=4.5, **kwargs):
    """Sepecially designed for TryOn."""
    codebook_emb_dim = 256
    codebook_size = 1024
    mask_token_id = model.mask_token_label
    _CONFIDENCE_OF_KNOWN_TOKENS = +np.inf    

    # encode img
    gt_z_q, gt_indices = get_encoded_results(model, batch['gt'])
    inpaint_z_q, inpaint_indices = get_encoded_results(model, batch['inpaint'])
    garment_z_q, garment_indices = get_encoded_results(model, batch['garment'])
    inpaint_warped_z_q, inpaint_warped_indices = get_encoded_results(model, batch['inpaint_warped'])

    # mask img
    # Defined the bounding box coordinates (xmin, ymin, xmax, ymax)
    bbox = batch['bbox']
    # z_q = inpaint_z_q 
    z_q = gt_z_q 
    # z_q = garment_z_q 
    patch_size = 16    
    batch_size = z_q.shape[0]
    latent_mask = torch.zeros((z_q.shape[0], z_q.shape[2], z_q.shape[3])).to(z_q.device)
    
    latent_t = torch.clamp(bbox[:, 1] // patch_size - 1, min=0)
    latent_b = torch.clamp(bbox[:, 3] // patch_size + 1, max=z_q.shape[2])
    latent_l = torch.clamp(bbox[:, 0] // patch_size - 1, min=0)
    latent_r = torch.clamp(bbox[:, 2] // patch_size + 1, max=z_q.shape[3])
    
    for i in range(batch_size):
        latent_mask[i, latent_t[i]:latent_b[i], latent_l[i]:latent_r[i]] = 1
    latent_mask = latent_mask.view(batch_size, -1)
    
    unknown_number_in_the_beginning = torch.sum(latent_mask).item()
    # masked_indices = inpaint_indices.clone()
    masked_indices = gt_indices.clone()
    # masked_indices = garment_indices.clone()
    masked_indices[latent_mask.nonzero(as_tuple=True)] = mask_token_id

    token_indices = masked_indices.clone()
    sampled_id_dict = {}
    sampled_id_dict['gt'] = gt_indices.squeeze().tolist()
    sampled_id_dict['inpaint'] = inpaint_indices.squeeze().tolist()
    sampled_id_dict['garment'] = garment_indices.squeeze().tolist()
    sampled_id_dict["step0:masked"] = token_indices.squeeze().tolist()
    sampled_loss_list = []

    # get garment indices 
    garment_indices = torch.cat([torch.zeros(garment_indices.size(0), 1).cuda(device=garment_indices.device), garment_indices], dim=1)
    if args.use_fake_class_label:
        garment_indices[:, 0] = model.fake_class_label 
    if args.use_random_class_label:
        garment_indices[:, 0] = random.randint(0, 1000)
    if args.use_customized_class_label:
        garment_indices[:, 0] = model.fake_class_label + 100
    if args.no_class_label:
        garment_indices = garment_indices[:, 1:]
    garment_indices = garment_indices.long()
    garment_embeddings = model.token_emb(garment_indices)

    # get inpaint_warped indices 
    inpaint_warped_indices = torch.cat([torch.zeros(inpaint_warped_indices.size(0), 1).cuda(device=inpaint_warped_indices.device), inpaint_warped_indices], dim=1)
    if args.use_fake_class_label:
        inpaint_warped_indices[:, 0] = model.fake_class_label 
    if args.use_random_class_label:
        inpaint_warped_indices[:, 0] = random.randint(0, 1000)
    if args.use_customized_class_label:
        inpaint_warped_indices[:, 0] = model.fake_class_label + 200
    if args.no_class_label:
        inpaint_warped_indices = inpaint_warped_indices[:, 1:]
    inpaint_warped_indices = inpaint_warped_indices.long()
    inpaint_warped_embeddings = model.token_emb(inpaint_warped_indices)

    use_ip_adapter = kwargs.get('use_ip_adapter', False)
    if use_ip_adapter:
        assert hasattr(model, 'ip_projector'), "Model does not have ip_projector attribute. Make sure IpAdapterWrapper.refactor_model() has been called."
    for step in range(num_iter):
        cur_ids = token_indices.clone().long()
        # sampled_id_dict[f"step{step}:masked"] = cur_ids.squeeze().tolist()

        token_indices = torch.cat(
            [torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        # TODO: add a random fake class label for testing. 
        # token_indices[:, 0] = torch.randint(0, 1100, (token_indices.size(0),)).cuda()
        token_indices[:, 0] = model.fake_class_label
        token_indices = token_indices.long()
        token_all_mask = token_indices == mask_token_id

        # no drop mask in inference
        token_drop_mask = torch.zeros_like(token_indices)

        # token embedding
        input_embeddings = model.token_emb(token_indices)

        # encoder
        x = input_embeddings
        bsz, seq_len, e_dim = x.shape
        if args.concat_garment:
            x = torch.cat([x, garment_embeddings], dim=1)
        if args.concat_inpaint_warped:
            x = torch.cat([x, inpaint_warped_embeddings], dim=1)
        
        if use_ip_adapter:
            # reshape the z_q to match the seq_len of the input embeddings
            vqgan_seq_len = inpaint_warped_z_q.shape[1]
            inpaint_warped_z_q = inpaint_warped_z_q.view(bsz, vqgan_seq_len, -1)
            garment_z_q = garment_z_q.view(bsz, vqgan_seq_len, -1)
            
            # Add a zero tensor at the head
            zero_tensor = torch.zeros(bsz, 1, inpaint_warped_z_q.size(-1), device=inpaint_warped_z_q.device)
            inpaint_warped_z_q = torch.cat([zero_tensor, inpaint_warped_z_q], dim=1)
            garment_z_q = torch.cat([zero_tensor, garment_z_q], dim=1)

            context = torch.cat([inpaint_warped_z_q, garment_z_q], dim=1)
            context = model.ip_projector(context)
            for blk in model.blocks:
                x = blk(x, context)
        else:
            for blk in model.blocks:
                x = blk(x)
        x = model.norm(x)
        # only keep the embeddings of the inpaint part
        x = x[:, :seq_len, :]
        # x = x[:, :, :e_dim]

        # decoder
        logits = model.forward_decoder(x, token_drop_mask, token_all_mask)
        logits = logits[:, 1:, :codebook_size]

        # get token prediction
        sample_dist = torch.distributions.categorical.Categorical(logits=logits)
        sampled_ids = sample_dist.sample()
        # sampled_id_dict[f"step{step}:sampled"] = sampled_ids.squeeze().tolist()

        # get ids for next step
        unknown_map = (cur_ids == mask_token_id)
        sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
        
        # get sample loss
        sample_stats = get_sample_stats(sampled_ids, cur_ids, gt_indices, logits, mask_token_id, step)
        sampled_loss_list.append(sample_stats)

        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / num_iter
        mask_ratio = np.cos(math.pi / 2. * ratio)

        # sample ids according to prediction confidence
        probs = torch.nn.functional.softmax(logits, dim=-1)
        selected_probs = torch.squeeze(
            torch.gather(probs, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)
        selected_probs = torch.where(unknown_map, selected_probs.double(), _CONFIDENCE_OF_KNOWN_TOKENS).float()

        mask_len = torch.Tensor([np.floor(unknown_number_in_the_beginning * mask_ratio)]).cuda()
        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                 torch.minimum(torch.sum(unknown_map, dim=-1, keepdims=True) - 1, mask_len))

        # Sample masking tokens for next iteration
        masking = mask_by_random_topk(mask_len, selected_probs, choice_temperature * (1 - ratio))

        # Masks tokens with lower confidence.
        token_indices = torch.where(masking, mask_token_id, sampled_ids)
    
    general_mask_accuracy = (sampled_ids[latent_mask.nonzero(as_tuple=True)] == gt_indices[latent_mask.nonzero(as_tuple=True)]).sum()/unknown_number_in_the_beginning
    total_tokens = sampled_ids.shape.numel()
    known_number_in_the_beginning = total_tokens - unknown_number_in_the_beginning
    kept_mask = ~latent_mask.bool()
    general_kept_accuracy = (sampled_ids[kept_mask] == gt_indices[kept_mask]).sum()/known_number_in_the_beginning
    
    sampled_id_dict[f"step{step}:sampled"] = sampled_ids.squeeze().tolist()
    sampled_loss_list.append({
        'general_mask_accuracy': general_mask_accuracy.item(),
        'general_kept_accuracy': general_kept_accuracy.item()
    })
    # vqgan visualization
    sampled_size = int(model.patch_embed.num_patches**0.5)
    z_q = model.vqgan.quantize.get_codebook_entry(sampled_ids, shape=(sampled_ids.shape[0], sampled_size, sampled_size, codebook_emb_dim))
    gen_images = model.vqgan.decode(z_q)
    return gen_images.view(bsz, 3, args.input_size, -1), {'loss_stats': sampled_loss_list, 'sampled_ids': sampled_id_dict}


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        data = np.load(path)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']

        return moments, target
