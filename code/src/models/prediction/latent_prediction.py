import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange, repeat
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid

from src.models.encoders.autoencoder import (AutoencoderKL, IdentityFirstStage)
from src.modules.distributions.distributions import (
    DiagonalGaussianDistribution, normal_kl)
from src.modules.ogm_metric.image_similarity_metric import ImageSimialrityMetric
from src.modules.ogm_metric.ogm_utils import convert_model_output_to_ogm
from src.util import count_params, instantiate_from_config

def gaussian_probs(x, mu=5, sigma=1):
    density = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)
    return density/np.sum(density)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

class LatentPrediction(pl.LightningModule):
    def __init__(self,
                 encoder_stage_config,
                 prediction_stage_config,
                 encoder_stage_key,
                 prediction_stage_key,
                 latent_stage_key,
                 loss_type="latent_mse_loss",
                 ckpt_path=None,
                 monitor="val/loss",
                 image_size=256,
                 channels=3,
                 obs_horizon=5,
                 future_horizon=15,
                 used_cached_latents=True,
                 log_every_n_steps=10000,
                 training_extrap_t = 5,
                 use_maps = False,
                 use_camera = False,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        print(f'LatentPrediction: Restarted from ckpt: {ckpt_path}.')
        print(f'Using maps: {use_maps}')
        if ckpt_path == 'None':
            ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        self.training_extrap_t = training_extrap_t
        self.encoder_stage_config = encoder_stage_config
        self.prediction_stage_config = prediction_stage_config
        self.encoder_stage_key = encoder_stage_key
        self.prediction_stage_key = prediction_stage_key
        self.latent_stage_key = latent_stage_key
        self.loss_type = loss_type
        self.image_size = image_size  
        self.channels = channels
        self.obs_horizon = obs_horizon
        self.future_horizon = future_horizon
        self.log_every_n_steps = log_every_n_steps
        self.used_cached_latents = used_cached_latents
        self.use_maps = use_maps
        self.use_camera = use_camera
        self.loss = self.get_loss(self.loss_type)


        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if not self.used_cached_latents:
            self.instantiate_encoder_stage(encoder_stage_config)
            count_params(self.encoder_stage_model, verbose=True)

        self.instantiate_prediction_stage(prediction_stage_config)
        count_params(self.prediction_stage_model, verbose=True)

        self.restarted_from_ckpt = False
        print(f'LatentPrediction: Restarted from ckpt: {ckpt_path}')
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
            
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def get_loss(self, loss_type):
        if self.loss_type == 'latent_mse_loss':
            return self.latent_mse_loss
        else:
            raise NotImplementedError(f"loss_type '{loss_type}' not implemented yet")

        return loss

    def load_data(self, batch, k):
        x = batch[k]
        if k.split('_')[0] != 'latent' and k.split('_')[-1] != 'maps' and k.split('_')[0] != 'CAM':
            if len(x.shape) == 3:
                x = x[..., None]
            
            if len(x.shape) == 5:
                x = rearrange(x, 'b t h w c -> b t c h w')
            else:
                x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, train=True)

        if self.global_step % self.log_every_n_steps == 0:
            ogm_metric_dict = self.log_metrics(batch)
            loss_dict = {**loss_dict, **ogm_metric_dict}

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        # if self.use_scheduler:
        #     lr = self.optimizers().param_groups[0]['lr']
        #     self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch, train=False)
        if batch_idx % 1000 == 0:
            print(f"Logging")
            ogm_metric_dict = self.log_metrics(batch)
            loss_dict_no_ema = {**loss_dict_no_ema, **ogm_metric_dict}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def instantiate_encoder_stage(self, config):
        model = instantiate_from_config(config)
        encoder_stage_model = model.eval()
        encoder_stage_model.train = disabled_train
        for param in encoder_stage_model.parameters():
            param.requires_grad = False
        
        if self.used_cached_latents:
            encoder_stage_model = encoder_stage_model.to('cpu')
        return encoder_stage_model

    def instantiate_prediction_stage(self, config):
        model = instantiate_from_config(config)
        self.prediction_stage_model = model.eval()

    def get_encoder_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.mode() #TODO: why
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return z 

    def predict(self, obs, future=None, 
                maps_z=None,
                cam_z=None,
                train=False, 
                return_loss=False,
                test=False):

        prefix = 'train' if self.training else 'val'

        if prefix == 'val':
            extrap_t = 5
        else:
            potential_t = np.arange(1, 19, 1)
            if self.training_extrap_t == 'random':
                print(f'Random')
                extrap_t = np.random.choice(range(1,19), p=gaussian_probs(potential_t, sigma=4))
            elif self.training_extrap_t == 'fixed':
                extrap_t = 5
            else:
                raise NotImplementedError(f"training_extrap_t of type '{self.training_extrap_t}' not yet implemented")
            
        net_input = torch.cat([obs, future], dim=1)
        num_epoch_masking = 20
        if self.use_maps:
            maps_input = maps_z
            probs = torch.linspace(1, 0.1, num_epoch_masking).numpy()
            if self.current_epoch < num_epoch_masking and train:
                masking = True
                prob_masking = probs[self.current_epoch]
            else:
                masking = False
                prob_masking = 0.0
        else:
            maps_input = None

        if self.use_camera:
            cam_input = cam_z
            probs = torch.linspace(1, 0.1, num_epoch_masking).numpy()
            if self.current_epoch < num_epoch_masking and train:
                masking = True
                prob_masking = probs[self.current_epoch]
            else:
                masking = False
                prob_masking = 0.0
        else:
            cam_input = None

        if not self.use_maps and not self.use_camera:
            masking = False
            prob_masking = 0.0

        if not train:
            masking = False
            prob_masking = 0.0
        prefix = 'train' if self.training else 'val'

        if maps_input is None and cam_input is None:
            pred, loss, loss_dict = self.prediction_stage_model(net_input, return_loss=return_loss, prefix=prefix, epoch_number=self.current_epoch, steps=self.global_step, test=test) 
        else:
            pred, loss, loss_dict = self.prediction_stage_model(net_input, maps_input=maps_input, camera_input=cam_input, 
                                                                return_loss=return_loss, prefix=prefix, epoch_number=self.current_epoch, steps=self.global_step, test=test,
                                                                 masking=masking, prob_masking=prob_masking)
        
        if loss is not None:
            assert loss_dict is not None and return_loss, f'Loss dict is None: {loss_dict is None} and return_loss: {return_loss}'
            out_loss_dict = {}
            for k, v in loss_dict.items():
                out_loss_dict[f'{prefix}/{k}'] = v
            return pred, loss, out_loss_dict
        else:
            return pred, None, None

    @torch.no_grad()
    def get_input(self, batch, obs_key, future_key, latent_key=None, bs=None, evaluation=False): #return_decoding=False, return_ground_truth=False):

        if evaluation:
            device = 'cpu'
        else:
            device = self.device

        outputs = []

        if evaluation:

            #Load the model
            encoder_stage_model = self.instantiate_encoder_stage(self.encoder_stage_config)

            obs = self.load_data(batch, obs_key)
            future = self.load_data(batch, future_key)
            if bs is not None:
                obs = obs[:bs]
                future = future[:bs]

            obs = obs.to(device)
            future = future.to(device)
            
            B, T_past, C, H, W = obs.shape
            B, T_future, C, H, W = future.shape

            #Encode all observations
            obs = obs.reshape(B*T_past, C, H, W)
            future = future.reshape(B*T_future, C, H, W)
            temp = torch.cat([obs, future], dim=0)

            encoder_posterior = self.encode_first_stage(temp, encoder_stage_model)
            temp = self.get_encoder_encoding(encoder_posterior).detach()
            
            obs_z, future_z = torch.split(temp, [obs.shape[0], future.shape[0]], dim=0)
            obs_z = rearrange(obs_z, '(b t) c h w -> b t c h w', b=B, t=T_past)
            future_z = rearrange(future_z, '(b t) c h w -> b t c h w', b=B, t=T_future)

            # obs, future, obs_z, future_z
            outputs.extend([obs_z.to(device), future_z.to(device)])

            future = future.reshape(B, T_future, C, H, W)
            obs = obs.reshape(B, T_past, C, H, W)
            outputs.extend([obs.to(device), future.to(device)])

            temp = torch.cat([obs_z, future_z], dim=1)
            temp = rearrange(temp, 'b t c h w -> (b t) c h w')
            temp = self.decode_first_stage(temp, encoder_stage_model=encoder_stage_model)
            temp = rearrange(temp, '(b t) c h w -> b t c h w', b=B, t=T_past+T_future)
            obs_rec, future_rec = torch.split(temp, [obs_z.shape[1], future_z.shape[1]], dim=1)
            outputs.extend([obs_rec.to(device), future_rec.to(device)])
            #return outputs # obs_z, future_z, obs, future, obs_rec, future_rec
        
  
        obs_z_cache = self.load_data(batch, f'{latent_key}_{obs_key}')
        future_z_cache = self.load_data(batch, f'{latent_key}_{future_key}')

        if self.use_maps:
            driveable_maps_z_cache = self.load_data(batch, f'driveable_maps')
            ped_cross_maps_z_cache = self.load_data(batch, f'ped_cross_maps')
            stop_line_maps_z_cache = self.load_data(batch, f'stop_line_maps')

        if self.use_camera:
            cam_f_z_cache = self.load_data(batch, f'CAM_FRONT')
            cam_fl_z_cache = self.load_data(batch, f'CAM_FRONT_LEFT')
            cam_fr_z_cache = self.load_data(batch, f'CAM_FRONT_RIGHT')
            cam_bl_z_cache = self.load_data(batch, f'CAM_BACK_LEFT')
            cam_br_z_cache = self.load_data(batch, f'CAM_BACK_RIGHT')
            cam_b_z_cache = self.load_data(batch, f'CAM_BACK')

        if bs is not None:
            obs_z_cache = obs_z_cache[:bs]
            future_z_cache = future_z_cache[:bs]  
            if self.use_maps:
                driveable_maps_z_cache = driveable_maps_z_cache[:bs]
                ped_cross_maps_z_cache = ped_cross_maps_z_cache[:bs]
                stop_line_maps_z_cache = stop_line_maps_z_cache[:bs]

            if self.use_camera:
                cam_f_z_cache = cam_f_z_cache[:bs]
                cam_fl_z_cache = cam_fl_z_cache[:bs]
                cam_fr_z_cache = cam_fr_z_cache[:bs]
                cam_bl_z_cache = cam_bl_z_cache[:bs]
                cam_br_z_cache = cam_br_z_cache[:bs]
                cam_b_z_cache = cam_b_z_cache[:bs]

        obs_z_cache = obs_z_cache.to(device)
        future_z_cache = future_z_cache.to(device)
        if self.use_maps:
            driveable_maps_z_cache = driveable_maps_z_cache.to(device)
            ped_cross_maps_z_cache = ped_cross_maps_z_cache.to(device)
            stop_line_maps_z_cache = stop_line_maps_z_cache.to(device)

        if self.use_camera:
            cam_f_z_cache = cam_f_z_cache.to(device)
            cam_fl_z_cache = cam_fl_z_cache.to(device)
            cam_fr_z_cache = cam_fr_z_cache.to(device)
            cam_bl_z_cache = cam_bl_z_cache.to(device)
            cam_br_z_cache = cam_br_z_cache.to(device)
            cam_b_z_cache = cam_b_z_cache.to(device)

        B, T_past = obs_z_cache.shape[:2]
        B, T_future = future_z_cache.shape[:2]
        if self.use_maps:
            B, T_maps, C, H, W = driveable_maps_z_cache.shape

        if self.use_camera:
            B, T_cam, C, H, W = cam_f_z_cache.shape

    

        C = obs_z_cache.shape[2]
        if C == 128:
            obs_z_cache = rearrange(obs_z_cache, 'b t c h w -> (b t) c h w')
            future_z_cache = rearrange(future_z_cache, 'b t c h w -> (b t) c h w')
        if self.use_maps:
            driveable_maps_z_cache = rearrange(driveable_maps_z_cache, 'b t c h w -> (b t) c h w')
            ped_cross_maps_z_cache = rearrange(ped_cross_maps_z_cache, 'b t c h w -> (b t) c h w')
            stop_line_maps_z_cache = rearrange(stop_line_maps_z_cache, 'b t c h w -> (b t) c h w')

        if self.use_camera:
            cam_f_z_cache = rearrange(cam_f_z_cache, 'b t c h w -> (b t) c h w')
            cam_fl_z_cache = rearrange(cam_fl_z_cache, 'b t c h w -> (b t) c h w')
            cam_fr_z_cache = rearrange(cam_fr_z_cache, 'b t c h w -> (b t) c h w')
            cam_bl_z_cache = rearrange(cam_bl_z_cache, 'b t c h w -> (b t) c h w')
            cam_br_z_cache = rearrange(cam_br_z_cache, 'b t c h w -> (b t) c h w')
            cam_b_z_cache = rearrange(cam_b_z_cache, 'b t c h w -> (b t) c h w')
        if C == 128:
            obs_encoder_posterior = DiagonalGaussianDistribution(obs_z_cache)
            future_encoder_posterior = DiagonalGaussianDistribution(future_z_cache)
        if self.use_maps:
            driveable_maps_encoder_posterior = DiagonalGaussianDistribution(driveable_maps_z_cache)
            ped_cross_maps_encoder_posterior = DiagonalGaussianDistribution(ped_cross_maps_z_cache)
            stop_line_maps_encoder_posterior = DiagonalGaussianDistribution(stop_line_maps_z_cache)

        if self.use_camera:
            cam_f_encoder_posterior = DiagonalGaussianDistribution(cam_f_z_cache)
            cam_fl_encoder_posterior = DiagonalGaussianDistribution(cam_fl_z_cache)
            cam_fr_encoder_posterior = DiagonalGaussianDistribution(cam_fr_z_cache)
            cam_bl_encoder_posterior = DiagonalGaussianDistribution(cam_bl_z_cache)
            cam_br_encoder_posterior = DiagonalGaussianDistribution(cam_br_z_cache)
            cam_b_encoder_posterior = DiagonalGaussianDistribution(cam_b_z_cache)
        if C == 128:
            obs_z_cache = self.get_encoder_encoding(obs_encoder_posterior).detach()
            future_z_cache = self.get_encoder_encoding(future_encoder_posterior).detach()
        if self.use_maps:
            driveable_maps_z_cache = self.get_encoder_encoding(driveable_maps_encoder_posterior).detach()
            ped_cross_maps_z_cache = self.get_encoder_encoding(ped_cross_maps_encoder_posterior).detach()
            stop_line_maps_z_cache = self.get_encoder_encoding(stop_line_maps_encoder_posterior).detach()

        if self.use_camera:
            cam_f_z_cache = self.get_encoder_encoding(cam_f_encoder_posterior).detach()
            cam_fl_z_cache = self.get_encoder_encoding(cam_fl_encoder_posterior).detach()
            cam_fr_z_cache = self.get_encoder_encoding(cam_fr_encoder_posterior).detach()
            cam_bl_z_cache = self.get_encoder_encoding(cam_bl_encoder_posterior).detach()
            cam_br_z_cache = self.get_encoder_encoding(cam_br_encoder_posterior).detach()
            cam_b_z_cache = self.get_encoder_encoding(cam_b_encoder_posterior).detach()
        if C == 128:
            obs_z_cache = rearrange(obs_z_cache, '(b t) c h w -> b t c h w', b=B, t=T_past)
            future_z_cache = rearrange(future_z_cache, '(b t) c h w -> b t c h w', b=B, t=T_future)
        if self.use_maps:
            driveable_maps_z_cache = rearrange(driveable_maps_z_cache, '(b t) c h w -> b t c h w', b=B, t=T_maps)
            ped_cross_maps_z_cache = rearrange(ped_cross_maps_z_cache, '(b t) c h w -> b t c h w', b=B, t=T_maps)
            stop_line_maps_z_cache = rearrange(stop_line_maps_z_cache, '(b t) c h w -> b t c h w', b=B, t=T_maps)

        if self.use_camera:
            cam_f_z_cache = rearrange(cam_f_z_cache, '(b t) c h w -> b t c h w', b=B, t=T_cam)
            cam_fl_z_cache = rearrange(cam_fl_z_cache, '(b t) c h w -> b t c h w', b=B, t=T_cam)
            cam_fr_z_cache = rearrange(cam_fr_z_cache, '(b t) c h w -> b t c h w', b=B, t=T_cam)
            cam_bl_z_cache = rearrange(cam_bl_z_cache, '(b t) c h w -> b t c h w', b=B, t=T_cam)
            cam_br_z_cache = rearrange(cam_br_z_cache, '(b t) c h w -> b t c h w', b=B, t=T_cam)
            cam_b_z_cache = rearrange(cam_b_z_cache, '(b t) c h w -> b t c h w', b=B, t=T_cam)

        outputs.extend([obs_z_cache.to(device), future_z_cache.to(device)])
        if self.use_maps:
            outputs.extend([driveable_maps_z_cache.to(device), ped_cross_maps_z_cache.to(device), stop_line_maps_z_cache.to(device)])

        if self.use_camera:
            outputs.extend([cam_f_z_cache.to(device), cam_fl_z_cache.to(device), cam_fr_z_cache.to(device), cam_b_z_cache.to(device)]) #cam_bl_z_cache.to(device), cam_br_z_cache.to(device),

        if evaluation:
            temp = torch.cat([obs_z_cache, future_z_cache], dim=1)
            temp = rearrange(temp, 'b t c h w -> (b t) c h w')
            temp = self.decode_first_stage(temp, encoder_stage_model)
            temp = rearrange(temp, '(b t) c h w -> b t c h w', b=B, t=T_past+T_future)
            obs_img_cache, future_img_cache = torch.split(temp, [obs_z_cache.shape[1], future_z_cache.shape[1]], dim=1)
            outputs.extend([obs_img_cache.to(device), future_img_cache.to(device), encoder_stage_model])
                
        return outputs
        
    @torch.no_grad()
    def decode_first_stage(self, z, encoder_stage_model):
        if encoder_stage_model is not None:
            return encoder_stage_model.decode(z)
        return self.encoder_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        return self.encoder_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x, encoder_stage_model=None):
        if encoder_stage_model is not None:
            return encoder_stage_model.encode(x)
        return self.encoder_stage_model.encode(x)
    
    
    def shared_step(self, batch, train, **kwargs):
        if not self.use_maps and not self.use_camera:
            obs_z, future_z = self.get_input(batch, self.encoder_stage_key, self.prediction_stage_key, self.latent_stage_key)
            loss = self(obs_z, future_z, train, train=train)
        else:
            obs_z, future_z, driveable_maps_z, ped_cross_maps_z, stop_line_maps_z, cam_f_z, cam_fl_z, cam_fr_z, cam_b_z = self.get_input(batch, self.encoder_stage_key, self.prediction_stage_key, self.latent_stage_key)
            cam_z = torch.cat([cam_f_z, cam_fl_z, cam_fr_z, cam_b_z], dim=1)
            maps_z = torch.cat([driveable_maps_z, ped_cross_maps_z, stop_line_maps_z], dim=1)
            loss = self(obs_z, future_z, 
                        maps_z,
                        cam_z,
                        train=train)
        return loss
    
    def predict_and_return_z(self, batch, train=False, test=False):
        if not self.use_maps and not self.use_camera:
            obs_z, future_z = self.get_input(batch, self.encoder_stage_key, self.prediction_stage_key, self.latent_stage_key)
            pred_z = self(obs_z, future_z, train, return_prediction=True, test=test)
        else:
            #cam_bl_z, cam_br_z, 
            obs_z, future_z, driveable_maps_z, ped_cross_maps_z, stop_line_maps_z, cam_f_z, cam_fl_z, cam_fr_z, cam_b_z = self.get_input(batch, self.encoder_stage_key, self.prediction_stage_key, self.latent_stage_key)
            cam_z = torch.cat([cam_f_z, cam_fl_z, cam_fr_z, cam_b_z], dim=1)
            maps_z = torch.cat([driveable_maps_z, ped_cross_maps_z, stop_line_maps_z], dim=1)
            pred_z = self(obs_z, future_z, 
                        maps_z,
                        cam_z,
                        train=train,
                        return_prediction=True,
                        test=test)
            

        return pred_z, future_z


    def forward(self, obs_z, future_z, 
                maps_z=None,
                cam_z=None,
                train=False,
                return_prediction=False,
                test=False
                ):

        if hasattr(self.prediction_stage_model, 'loss') and callable(getattr(self.prediction_stage_model, 'loss')):
            # The class model has a method 'loss'
            # Add your code here
        
            pred_z, loss, loss_dict = self.predict(obs_z, future_z, 
                                  maps_z,
                                  cam_z,
                                  train=train,
                                  return_loss=True,
                                  test=test)
            if return_prediction:
                return pred_z
            return loss, loss_dict
        else:
            pred_z = self.predict(obs_z, future_z, 
                                  maps_z,
                                  cam_z,
                                  train=train,
                                  test=test)
            if return_prediction:
                return pred_z
            return self.loss(pred_z, future_z, obs_z)



    def predict_and_decode(self, obs_z, return_latents=False, encoder_stage_model=None):
        pred = self.predict(obs_z.to(self.device))
        pred = pred.to('cpu')
        B = pred.shape[0]
        T = pred.shape[1]
        pred = rearrange(pred, 'b t c h w -> (b t) c h w') 
        decoded_pred = self.decode_first_stage(pred, encoder_stage_model)
        decoded_pred = decoded_pred.reshape(B, T, decoded_pred.shape[-3], decoded_pred.shape[-2], decoded_pred.shape[-1])
        if return_latents:
            pred = pred.reshape(B, T, pred.shape[-3], pred.shape[-2], pred.shape[-1])
            return decoded_pred, pred
        return decoded_pred

    @torch.no_grad()
    def log_metrics(self, batch):
        prefix = 'train' if self.training else 'val'
        log = dict()
        return log

    @torch.no_grad()
    def log_images(self, batch, N=6, n_row=4, **kwargs):
        log = dict()
        return log
    
    def latent_mse_loss(self, pred, pred_gt, obs):
        """Loss between latents"""
        prefix = 'train' if self.training else 'val'
        if prefix == 'train':
            pred_gt = torch.concat([obs, pred_gt], dim=1)
            optimized_loss = ((pred - pred_gt[:,1:])**2) #((pred[:,-15:] - pred_gt[:,-15:])**2) 
            loss_org = ((pred[:,-15:] - pred_gt[:,-15:])**2)
            
        else:
            pred_gt = torch.concat([obs, pred_gt], dim=1)
            optimized_loss = ((pred - pred_gt[:,1:])**2) #((pred[:,-15:] - pred_gt[:,-15:])**2) 
            loss_org = ((pred[:,-15:] - pred_gt[:,-15:])**2)
        
        optimized_loss = optimized_loss.mean()
        loss_org = loss_org.mean()

        loss_dict = {
            f'{prefix}/loss_mse_latents': loss_org,
            f'{prefix}/optimized_loss': optimized_loss,
        }

        return optimized_loss, loss_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        params = params + list(self.prediction_stage_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
    
    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid
