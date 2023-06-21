import os
import pickle
import cv2
import torch
import yaml
from torch.utils.data import Dataset, Subset
import json



class Base(Dataset):
    def __init__(self, path, size):
        self.path = path
        self.size = size
        self._prepare()

    def _prepare(self):
        self.samples = []
        for scene in os.listdir(self.path):
            path_to_scene = os.path.join(self.path, scene)
            for sensor in os.listdir(path_to_scene):
                if sensor.endswith('.png'):
                    path_to_sensor = os.path.join(path_to_scene, sensor)
                    self.samples.append(path_to_sensor)

    def __len__(self):
        return len(self.samples)
    
    def _load_img(self, path):
        try:
            if not path.endswith('.png'):
                path += '.png'
            
            cur_s = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            cur_s = cv2.cvtColor(cur_s, cv2.COLOR_BGR2RGB)

            cur_s = cur_s / 255.0

            if cur_s.shape[1] != self.size and cur_s.shape[2] != self.size:
                cur_s = cv2.resize(cur_s, (self.size,  self.size))
            s_t = cur_s.astype('float32')
            s_t = torch.FloatTensor((s_t - 0.5) / 0.5)
            return s_t
        except:
            print(f'Error loading image: {path}. \n')
            return None
    
    def __getitem__(self, i):
        fn = os.path.join(self.path,self.samples[i])   
        img = self._load_img(fn)
        sample = {
            'filename': fn,
            'image': img,
        }
        return sample
 
    
class Seq(Base):
    def __init__(self, path_ogm_ras, path_ogm_latents, 
                       past_len, future_len, 
                       step, size, 
                       use_maps=False, path_map_ras=None, path_map_latents=None,
                       use_cameras=False, path_cam_ras=None, path_cam_latents=None,
                       return_ras_ogm=False, return_ras_map=False, return_ras_cam=False, 
                       mode='all'):
        self.path_ogm_ras = path_ogm_ras
        self.path_ogm_latents = path_ogm_latents
        self.past_len = past_len
        self.future_len = future_len
        self.nt = past_len + future_len
        self.step = step
        self.mode = mode
        self.size = size
        self.use_maps = use_maps
        self.use_cameras = use_cameras
        self.return_ras_map = return_ras_map
        self.return_ras_cam = return_ras_cam
        self.return_ras_ogm = return_ras_ogm

        if self.use_maps:
            self.map_types = ['driveable', 'ped_cross', 'stop_line']
            self.path_map_ras = path_map_ras
            self.path_maps_latents = path_map_latents

        if self.use_cameras:
            self.camera_types = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
            self.path_cam_ras = path_cam_ras
            self.path_cam_latents = path_cam_latents
            self.sync_path = None
        
        self._prepare_seq()

    def _prepare_seq(self):
        self.possible_starts = []

        self.scenes = os.listdir(self.path_ogm_ras)
        self.scene_len = {}
                
        #Measure the length of the scene.
        for scene in self.scenes:
            frames = os.listdir(os.path.join(self.path_ogm_ras, scene))
            self.scene_len[scene] = max([int(frame.split('.')[0]) for frame in frames])

        #Identify possible_starts
        possible_starts = []

        for scene in self.scene_len.keys():
            _len = self.scene_len[scene]

            if self.mode == 'all':
                starts = list(range(0,_len-self.nt*self.step, self.step))
            elif self.mode == 'unique':
                starts = list(range(0,_len-self.nt*self.step,self.step*self.nt))

            for idx in starts:
                possible_starts.append((scene,idx))
            self.possible_starts = possible_starts

        print(f'Length of the dataset: {len(self.possible_starts)}')

    def __len__(self):
        return len(self.possible_starts)
    
    def __getitem__(self, i):

        scene, start_idx = self.possible_starts[i]
        past_end_idx = start_idx + self.step*self.past_len
        end_idx = start_idx + self.step*self.nt 
        ogm_grids = []
        ogm_latents = []

        if self.use_maps:
            map_dict = {
                'driveable_latent': [],
                'ped_cross_latent': [],
                'stop_line_latent': [],
            }

            if self.return_ras_map:
                map_dict['driveable_ras'] = []
                map_dict['ped_cross_ras'] = []
                map_dict['stop_line_ras'] = []
        
        if self.use_cameras:
            with open(os.path.join(self.sync_path, scene + '_sync')) as f:
                sync_data = json.load(f)   
     
            camera_dict = {
                'CAM_FRONT_latent': [],
                'CAM_FRONT_LEFT_latent': [],
                'CAM_FRONT_RIGHT_latent': [],
                'CAM_BACK_LEFT_latent': [],
                'CAM_BACK_RIGHT_latent': [],
                'CAM_BACK_latent': [],
            }

            if self.return_ras_cam:
                camera_dict['CAM_FRONT_ras'] = []
                camera_dict['CAM_FRONT_LEFT_ras'] = []
                camera_dict['CAM_FRONT_RIGHT_ras'] = []
                camera_dict['CAM_BACK_LEFT_ras'] = []
                camera_dict['CAM_BACK_RIGHT_ras'] = []
                camera_dict['CAM_BACK_ras'] = []
        
        for loc in range(start_idx, end_idx, self.step):
            ras_filename = os.path.join(self.path_ogm_ras, str(scene), str(loc) + '.png')
            latent_filename = os.path.join(self.path_ogm_latents, str(scene), str(loc) + '.pkl')
            if True: # loc == start_idx + self.step*(self.past_len-1): # loc % 20 == 0
                if self.use_maps:
                    for map_type in self.map_types:
                        # print(f'Map filname elements:{self.path_maps_latents, str(scene), str(loc) + f"_{map_type}.pkl"}')
                        map_latent_filename = os.path.join(self.path_maps_latents, str(scene), str(loc) + f'_{map_type}.pkl')
                        latent_map = self.load_pickle(map_latent_filename)
                        map_dict[f'{map_type}_latent'].append(latent_map)

                        if self.return_ras_map:
                            map_ras_filename = os.path.join(self.path_map_ras, str(scene), str(loc) + f'_{map_type}.png')
                            map_ras = self._load_img(map_ras_filename)
                            map_dict[f'{map_type}_ras'].append(map_ras)

                if self.use_cameras:
                    cam_timesteps = sync_data[loc]
                    for camera_type in self.camera_types:
                        t_cam = cam_timesteps[camera_type]
                        camera_latent_filename = os.path.join(self.path_cam_latents, str(scene), f'{camera_type}_{t_cam}.pkl')
                        latent_camera = self.load_pickle(camera_latent_filename)
                        camera_dict[f'{camera_type}_latent'].append(latent_camera)

                        if self.return_ras_cam:
                            camera_ras_filename = os.path.join(self.path_cam_ras, str(scene), f'{camera_type}_{t_cam}.png')
                            camera_ras = self._load_img(camera_ras_filename)
                            camera_dict[f'{camera_type}_ras'].append(camera_ras)

            ogm_latent = self.load_pickle(latent_filename)
            if self.return_ras_ogm:
                ogm_ras = self._load_img(ras_filename)
                ogm_grids.append(ogm_ras)

            ogm_latents.append(ogm_latent)

        output = {
            'filenames': [scene, start_idx],
            'latent_observations': torch.concat(ogm_latents[:self.past_len], dim=0),
            'latent_future': torch.concat(ogm_latents[self.past_len:], dim=0),
        }

        if self.return_ras_ogm:
            output['ras_observations'] = torch.stack(ogm_grids[:self.past_len], dim=0)
            output['ras_future'] = torch.stack(ogm_grids[self.past_len:], dim=0)

        if self.use_maps:
            output['driveable_latent'] = torch.concat(map_dict['driveable_latent'], dim=0)
            output['ped_cross_latent'] = torch.concat(map_dict['ped_cross_latent'], dim=0)
            output['stop_line_latent'] = torch.concat(map_dict['stop_line_latent'], dim=0)

            if self.return_ras_map:
                output['driveable_ras'] = torch.stack(map_dict['driveable_ras'], dim=0)
                output['ped_cross_ras'] = torch.stack(map_dict['ped_cross_ras'], dim=0)
                output['stop_line_ras'] = torch.stack(map_dict['stop_line_ras'], dim=0)

        if self.use_cameras:
            output['CAM_FRONT_latent'] = torch.concat(camera_dict['CAM_FRONT_latent'], dim=0)
            output['CAM_FRONT_LEFT_latent'] = torch.concat(camera_dict['CAM_FRONT_LEFT_latent'], dim=0)
            output['CAM_FRONT_RIGHT_latent'] = torch.concat(camera_dict['CAM_FRONT_RIGHT_latent'], dim=0)
            output['CAM_BACK_LEFT_latent'] = torch.concat(camera_dict['CAM_BACK_LEFT_latent'], dim=0)
            output['CAM_BACK_RIGHT_latent'] = torch.concat(camera_dict['CAM_BACK_RIGHT_latent'], dim=0)
            output['CAM_BACK_latent'] = torch.concat(camera_dict['CAM_BACK_latent'], dim=0)

            if self.return_ras_cam:
                output['CAM_FRONT_ras'] = torch.stack(camera_dict['CAM_FRONT_ras'], dim=0)
                output['CAM_FRONT_LEFT_ras'] = torch.stack(camera_dict['CAM_FRONT_LEFT_ras'], dim=0)
                output['CAM_FRONT_RIGHT_ras'] = torch.stack(camera_dict['CAM_FRONT_RIGHT_ras'], dim=0)
                output['CAM_BACK_LEFT_ras'] = torch.stack(camera_dict['CAM_BACK_LEFT_ras'], dim=0)
                output['CAM_BACK_RIGHT_ras'] = torch.stack(camera_dict['CAM_BACK_RIGHT_ras'], dim=0)
                output['CAM_BACK_ras'] = torch.stack(camera_dict['CAM_BACK_ras'], dim=0)

        return output
    
    def load_pickle(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return torch.from_numpy(data).float()
