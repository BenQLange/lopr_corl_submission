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

<!-- ![](media/pred_1.gif)
![](media/pred_4.gif) -->
