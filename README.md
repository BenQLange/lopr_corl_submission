# LOPR: Latent Occupancy PRediction using Generative Models

This is the implementation of LOPR used in "LOPR: Latent Occupancy PRediction using Generative Models". In this repo, we provide the implementation and the visualization of our predictions.

## Abstract:
Environment prediction frameworks are integral for autonomous vehicles, enabling safe navigation in dynamic environments. Prior approaches have used occupancy grid maps as bird's eye-view representations of the scene, optimizing prediction architectures directly in grid cell space. While these methods have achieved some degree of success in spatiotemporal prediction, they occasionally grapple with unrealistic and incorrect predictions. 
We claim that the quality and realism of the forecasted occupancy grids can be enhanced with the use of generative models. We propose a framework that decouples occupancy prediction into two parts: representation learning and stochastic prediction within the learned latent space. 
Our approach allows for conditioning the model on other commonly available sensor modalities such as RGB-cameras and high definition~(HD) maps. We demonstrate that our approach achieves state-of-the-art performance and is readily transferable between different robotic platforms on the real-world NuScenes, Waymo Open, and our custom robotic datasets.

## Training
We provide the implementation details in the `code` directory. Before running each script, don't forget to update the corresponding paths pointing to the datasets and saved models.

Run the following script to train autoencoder for each sensor modality:
```
python main.py --base configs/autoencoder/autoencoder_4x4x64.yaml -t --gpus N 
```
Convert each dataset to the latent space running the following script:
```
python scripts/convert_to_latent_dataset.py
```
Run the following script to train the prediction network:
```
python main.py --base configs/prediction/variational_transformer_4x4x64.yaml -t --gpus N 
```

## Visualization on of the predictions

Each scene consists of observation (0.5s), ground truth future (1.5), and 20 randomly sampled predictions (1.5s). Each grid's cell contains a continuous value between 0 and 1 representing a probabilty of occupancy. Ground truth grids are generated with lidar sensor measurements. As a result, the dataset doesn't require any manual labelling and the prediction network can be trained in the self-supervised fashion. However, contrary to vectorized occupancy grid approaches that relies on the object detection framework, the grids are significantly noisier due to the stochasticity in the lidar's ray hits with objects, random reflections, and myriad of other small objects (trees, leaves, barriers, curbs, etc.). Below, we visualize couple of examples of predictions from Nuscenes Dataset that captures challenging scenarios due to the unknwon intent of the agent, occlusions, and partial observability.

Scene 1: Ego vehicle moving to the left is surrounded by other traffic particpants. Traffic above the ego vehicle is moving in the opposing direction to the right side of the grid. In the observed grids, the bus located above the ego vehicle is occluding majority of the environment. Our framework realistically forecasts potential futures of observed agents and reasons over the plausible occluded environments and other agents entering the scenes. For example, our framework infers: potential intersection  (Samples 2), potential oncoming agents following the bus similar to the ground truth  (Samples 3, 4, 12), parked cars that are occluded in the observed grids by the bus (Samples 6, 13, 14, 19, 20), and an empty straight road (Samples 5, 8, 10, 15, 16).
![](visualization/LOPR_GIF_618_210.gif)

Scene 2: Ego vehicle moving to the left is surrounded by other traffic particpants. Traffic above the ego vehicle is moving in the opposing direction to the right side of the grid. In the observed grids, the vehicle is passing our ego vehicle and moving to the right. In the ground truth future, the passing vehicle is immediately followed by another vehicle. Our framework realistically forecasts potential futures of observed agents and reasons over the plausible occluded environments and other agents entering the scenes. For example, our framework infers: another vehicle following the observed passing vehicle similar to the ground truth future (Sample 13), a vehicle parked passing below our ego vehicle (Samples 1, 7), potential intersection (Sample 4), and a realtively empty road (in all other samples).
![](visualization/LOPR_GIF_81_140.gif)

Scene 3: Ego vehicle turning at the intersection surroudned by traffic surrounded by other traffic particiapnts. Our framework realistically forecasts potential futures of observed agents and reasons over the plausible occluded environments and other agents entering the scenes. 
![](visualization/LOPR_GIF_490_140.gif)

