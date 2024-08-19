# On the Vulnerability of Skip Connection to Model Inversion Attacks (Plug & Play Attacks)

# Setup and Run Attacks

## GPU Memory Requirements
To provide a reference point for the GPU memory needed to perform the attacks, we measured the memory consumption for attacks against a ResNet-18 model trained on images with size 224x224. The memory consumption mainly depends on the batch size (BS), the target model size, and the StyleGAN2 model size. For our paper, we used V100-SXM3-32GB-H GPUs, but GPUs with smaller memory sizes are also sufficient.

| StyleGAN2 Resolution      | ResNet-18 (BS 20) | ResNet-18 (BS 10) | ResNet-18 (BS 1) |
| ----------- | ----------- | ----------------------- | ---------------- |
| 1024x1024   | 24.1 GB     | 15.8 GB                 | 3.2 GB           |
| 512x512     | 16.3 GB     | 10.8 GB                 | 2.7 GB           |
| 256x256     |  7.7 GB     |  5.1 GB                 | 2.1 GB           |


## Setup StyleGAN2
For using our attacks with StyleGAN2, clone the official [StyleGAN2-ADA-Pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repo into the project's root folder and remove its git specific folders and files. 
```
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
rm -r --force stylegan2-ada-pytorch/.git/
rm -r --force stylegan2-ada-pytorch/.github/
rm --force stylegan2-ada-pytorch/.gitignore
```

To download the pre-trained weights, run the following command from the project's root folder or copy the weights into ```stylegan2-ada-pytorch```:
```bash
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -P stylegan2-ada-pytorch/

```
NVIDIA provides the following pre-trained models: ```ffhq.pkl, metfaces.pkl, afhqcat.pkl, afhqdog.pkl, afhqwild.pkl, cifar10.pkl, brecahad.pkl```. Adjust the command above accordingly. For the training and resolution details, please visit the official repo.


## Prepare Datasets
We support [FaceScrub](http://vintage.winklerbros.net/facescrub.html) and [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) as datasets to train the target models. Please follow the instructions on the websites to download the datasets. Place all datasets in the folder ```data``` and make sure that the following structure is kept:

    .
    ├── data       
        ├── facescrub
            ├── actors
                ├── faces
                └── images
            └── actresses
                ├── faces
                └── images
        ├── stanford_dogs
            ├── Annotation
            ├── Images
            ├── file_list.mat
            ├── test_data.mat
            ├── test_list.mat
            ├── train_data.mat
            └── train_list.mat


            
## Train Target Models
Our code currently allows training of a wide range of deep neural networks including ResNet18, ResNet34, ResNet50, ResNet101, DenseNet121, DenseNet161, DenseNet169, DenseNet201 and MaxViT-T.

To define the model and training configuration, you need to create a configuration file. We provide a example configuration with explanations at ```configs/training/default_training.yaml```. For each model, create a separate configuration file for the full model and "skip-n removed" model. We provide examples of these configuration files in the `configs/training` folder for the various supported models.

To navigate the configs folder, we kept the file structure as shown below:

    .
    ├── configs       
        ├── attacking
            ├── DATASET
                ├── MODEL
                    ├── full.yaml
                    └── skip.yaml
        ├── training
            ├── evaluation
            ├── target
                ├── DATASET
                    ├── MODEL
                        ├── full.yaml
                        └── skip.yaml

To train the specified model, we provide a script to train the model easily. Run the following command to start the training with the specified configuration:

``` bash scripts/train_classifiers.sh```

For more infomation on how to use the script, please refer to the instructions available in the bash script file.

## Perform Attacks Using Checkpoints
To perform our attacks using the checkpoints provided, place all checkpoints in the folder ```results``` and make sure that the checkpoints are placed in their respective dataset subfolders. The resulting file structure should look like:

    .
    ├── results       
        ├── CelebA
            ├── <MODEL_ARCHITECTURE>
                ├── checkpoint.pth

        ├── FaceScrub
            ├── <MODEL_ARCHITECTURE>
                ├── checkpoint.pth
                
        ├── Stanford_Dogs
            ├── <MODEL_ARCHITECTURE>
                ├── checkpoint.pth
        
We provide an example configuration with explanations at ```configs/attacking/default_attacking.yaml```. We also provide configuration files to reproduce the various attack results stated in our paper. You only need to adjust the run paths for each dataset combination, and possibly the batch size.

To attack the specified model, we provide a script for MI attack. Run the following command to start the MI attack process with the specified configuration files:

``` bash scripts/attack.sh ```

For more infomation on how to use the script, please refer to the instructions available in the bash script file.

## Visualisation of Reconstructed Images
To visualise the inverted images after the attack process, we provide a python file `viz.py` that converts the filtered latents to RGB images. To run this python file, run the following script:

``` bash scripts/viz.sh ```

Within the bash script file, please specify the path to the model's configuration file (.yaml) and the filtered latent file (.pt). For example:

``` python viz.py -c <PATH/TO/CONFIG/FILE> -w <PATH/TO/LATENT/FILE> ```

