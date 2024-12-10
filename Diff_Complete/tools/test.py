import logging
import os
import random
import shutil
import warnings

import mcubes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.distributed import all_gather, get_world_size, is_master_proc
from lib.utils import AverageMeter, Timer
from lib.visualize import visualize_mesh
from models.diffusion import initialize_diff_model, load_diff_model
from models.diffusion.gaussian_diffusion import get_named_beta_schedule
from skimage.measure import marching_cubes
from PIL import Image


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

def test(model, control_model, data_loader, config, if_val = False):


  is_master = is_master_proc(config.exp.num_gpus) if config.exp.num_gpus > 1 else True
  cur_device = torch.cuda.current_device()
  global_timer, iter_timer = Timer(), Timer()

  bs = config.test.test_batch_size // config.exp.num_gpus

  model.eval()
  if control_model != None:
      control_model.eval()

  if is_master:
      logging.info('===> Start testing')
  global_timer.tic()

  # Clear cache (when run in test mode, cleanup training cache)
  torch.cuda.empty_cache()

  # Split test data into different gpus
  total_test_cnt = len(data_loader) // config.exp.num_gpus
  if if_val:
      # Randomly sample 3 indices from the range of possible test counts
      test_indices = random.sample(range(total_test_cnt), min(1, total_test_cnt))
      test_cnt = len(test_indices)
  else:
      test_cnt = total_test_cnt
      test_indices = range(test_cnt)

  test_iter = 0
  if control_model is not None:
      test_iter = int(config.net.control_weights[:-4].split('iter')[1])

  cls = config.data.class_id
  save_folder = 'completion_results'
  os.makedirs(save_folder, exist_ok=True)
  save_folder = os.path.join(save_folder, str(cls), str(test_iter))
  os.makedirs(save_folder, exist_ok=True)
  noise_folder = os.path.join(save_folder, 'noise')
  os.makedirs(noise_folder, exist_ok=True)

  npz_folder = os.path.join('completion_results_npz', str(cls), str(test_iter))
  os.makedirs(npz_folder, exist_ok=True)

  # Setting of Diffusion Models
  clip_noise = config.test.clip_noise
  use_ddim = config.test.use_ddim
  ddim_eta = config.test.ddim_eta
  betas = get_named_beta_schedule(config.diffusion.beta_schedule,
                                  config.diffusion.step,
                                  config.diffusion.scale_ratio)
  DiffusionClass = load_diff_model(config.diffusion.test_model)
  diffusion_model = initialize_diff_model(DiffusionClass, betas, config)

  data_iter = data_loader.__iter__()

  iter_timer.tic()

  # Add metrics tracking for validation
  if if_val:
      val_loss_meter = AverageMeter()
      val_score_meter = AverageMeter()

  if config.exp.num_gpus == 1:
      
      with torch.no_grad():
          for idx in test_indices:
              scan_ids, rendered_images, observe, gt = next(data_iter)
              sign = observe[:, 1].numpy()
              bs = observe.size(0)
              noise = None
              model_kwargs = {
                  'noise_save_path': os.path.join(noise_folder, f'{scan_ids[0]}noise.pt')}
              model_kwargs["hint"] = observe.to(cur_device) # torch.Size([1, 2, 32, 32, 32])
              model_kwargs["image"] = rendered_images.numpy()

              # Create sample-specific folder
              sample_folder = os.path.join(save_folder, scan_ids[0])
              os.makedirs(sample_folder, exist_ok=True)

              # Save input range scan visualization
              for i in range(len(observe)):
                  single_observe = observe[i]
                  obs_sdf = single_observe[0].numpy()
                  scan_id = scan_ids[i]
                  sdf_vertices, sdf_traingles = mcubes.marching_cubes(obs_sdf, 0.5)
                  out_file = os.path.join(sample_folder, f'input_{i}.obj')
                  mcubes.export_obj(sdf_vertices, sdf_traingles, out_file)

              # Save ground truth mesh
              gt_df = gt.numpy()
              for i in range(len(gt_df)):
                  gt_single = gt_df[i]
                  vertices, traingles = mcubes.marching_cubes(gt_single, 0.5)
                  out_file = os.path.join(sample_folder , f'gt_{i}.obj')
                  mcubes.export_obj(vertices, traingles, out_file)

              # Save rendered image input
              for i in range(len(rendered_images)):
                  for j in range(8): # Iterate through all 8 renderings for each sample
                    img = rendered_images[i, j]
                    img_path = os.path.join(sample_folder, f'input_render_{i}_{j}.png')
                    # Convert numpy array to PIL Image and save
                    Image.fromarray((img.numpy() * 255).astype(np.uint8)).save(img_path)
                #   img = rendered_images[i]
                #   img_path = os.path.join(sample_folder,f'input_render_{i}.png')
                #   # Convert numpy array to PIL Image and save
                #   Image.fromarray((img.numpy()).astype(np.uint8)).save(img_path)

              if use_ddim:
                  low_samples = diffusion_model.ddim_sample_loop(model=model,
                                                                 shape=[bs, 1] + [config.exp.res] * 3,
                                                                 device=cur_device,
                                                                 clip_denoised=clip_noise, progress=True,
                                                                 noise=noise,
                                                                 eta=ddim_eta,
                                                                 model_kwargs=model_kwargs).detach()
              else:
                  low_samples = diffusion_model.p_sample_loop(model=model,
                                                            control_model=control_model if control_model is not None else None,
                                                            shape=[bs, 1] + [config.exp.res] * 3,
                                                            device=cur_device,
                                                            clip_denoised=clip_noise, progress=True, noise=noise,
                                                            model_kwargs=model_kwargs).detach()

              low_samples = low_samples.cpu().numpy()[:, 0]
              if config.data.log_df == True:
                  low_samples = np.exp(low_samples) - 1
              low_samples = np.clip(low_samples, 0, config.data.trunc_distance)

              # Save predicted mesh in sample folder
              for i in range(len(low_samples)):
                  low_sample = low_samples[i]
                  vertices, traingles = mcubes.marching_cubes(low_sample, 0.5)
                  out_file = os.path.join(sample_folder, 'output.obj')
                  mcubes.export_obj(vertices, traingles, out_file)
                  out_npy_file = os.path.join(sample_folder, 'output.npy')
                  np.save(out_npy_file, low_sample)

              if if_val:
                  # Calculate validation metrics
                  # MSE loss between predicted samples and ground truth
                  pred_samples = torch.from_numpy(low_samples).to(cur_device)
                  gt_samples = gt.to(cur_device)
                  mse_loss = F.mse_loss(pred_samples, gt_samples)
                  val_loss_meter.update(mse_loss.item(), bs)
                  
                  # Calculate some score metric (e.g., PSNR or custom metric)
                  # This is just an example - adjust based on your needs
                  score = -mse_loss.item()  # Higher score for lower loss
                  val_score_meter.update(score, bs)

  else:
      with torch.no_grad():
          for scan_id, observe, gt in data_loader:
              sign = observe[:, 1].numpy()
              noise = None
              model_kwargs = {
                  'noise_save_path': os.path.join(noise_folder, f'{scan_id[0]}noise.pt')}
              model_kwargs["hint"] = observe.to(cur_device)  # torch.Size([1, 2, 32, 32, 32])

              if use_ddim:
                  low_samples = diffusion_model.ddim_sample_loop(model=model,
                                                                 shape=[bs, 1] + [config.exp.res] * 3,
                                                                 device=cur_device,
                                                                 clip_denoised=clip_noise, progress=True,
                                                                 noise=noise,
                                                                 eta=ddim_eta,
                                                                 model_kwargs=model_kwargs).detach()
              else:
                  low_samples = diffusion_model.p_sample_loop(model=model,
                                                              control_model=control_model if control_model is not None else None,
                                                              shape=[bs, 1] + [config.exp.res] * 3,
                                                              device=cur_device,
                                                              clip_denoised=clip_noise, progress=True, noise=noise,
                                                              model_kwargs=model_kwargs).detach()

              low_samples = low_samples.cpu().numpy()[:, 0]
              if config.data.log_df == True:
                  low_samples = np.exp(low_samples) - 1
              low_samples = np.clip(low_samples, 0, config.data.trunc_distance)

              # You can visualize the results here

  iter_time = iter_timer.toc(False)
  global_time = global_timer.toc(False)

  # Return validation metrics if in validation mode
  if if_val:
      return val_loss_meter.avg, val_score_meter.avg, {}
  return None, None, {}
