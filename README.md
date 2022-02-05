# Pixel-wise Anomaly Detection in Complex Driving Scenes
This repository is the PyTorch implementation of the paper, ["Pixel-wise Anomaly Detection in Complex Driving Scenes"](https://arxiv.org/abs/2103.05445). It is well documented version of the original repository with the code flow available [here](https://github.com/giandbt/synboost). The paper address the problem of anomaly segmentation.

![Alt text](display_images/methodology.png?raw=true "Methodology")

### Requirements

1) To install the requirements:
```pip install -r requirements_demo.txt```

2) Wandb has been used to log files and data, so it is necessary to have a wandb account (https://wandb.ai/site), details of which need to be added in the config file (`image_dissimilarity/configs/train/default_configuration.yaml`). 


### Pretrained Weights

1) Dissimilarity Module : [Link(dissimilarity models)](https://drive.google.com/drive/folders/16ELWb4Qu0AZ5dolf1vT5SoIkpdNR59DR?usp=sharing)
2) Segmentation and Resynthesis Models: [Link(Segmentation and resynthesis models)](https://drive.google.com/drive/folders/1OLsxpM_D6c8kGxikZwlytYDYZtR8S3C-?usp=sharing)


### Datasets 
The repository uses the Cityscapes Dataset [4] as the basis of the training data for the dissimilarity model. 
To download the dataset please register and follow the instructions here: https://www.cityscapes-dataset.com/downloads/.

Then, we need to pre-process the images in order to get the predicted entropy, distance, perceptual difference, synthesis, and semantic maps.
For this please provide the original cityscapes instanceids, labelids and the original images. Add the paths to these directories in the test_options.py under options folder.
Also provide pretrained weights of segmentation and synthesis.

For making the data, run the following:

1) For making the known data (i.e. data made using semantic predictions).
   ```
   python create_known_data.py --results_dir <path to results directory> --demo-folder <path to original images>
   ```

2) For making the unknown data (i.e. data that uses the ground truth semantic maps).
   ```
   python create_unknown_data.py --results_dir <path to results directory> --demo-folder <path to original images> --instances_og <path to original instance ids> --semantic_og <path to ground truth semantic maps>
   ```

- This is the link to dataset of full framework (provided in the original author's repo) : [Link(original)](http://robotics.ethz.ch/~asl-datasets/Dissimilarity/data_processed.tar)
- Link to dataset of light data : [Link(light data)](https://www.kaggle.com/mlrc2021anonymous/synboost-light-data)
- Link to dataset of w/o data generator: [Link(without datagenerator)](https://www.kaggle.com/mlrc2021anonymous/synboost-without-data-generator)

(For main model and light framework you can find labels in the Dataset)


### Training 
In order to train the dissimilarity network, modify the necessary parameters in the configuration file `image_dissimilarity/configs/train/default_configuration.yaml`. 
   - For w/o uncertainty maps make prior = false
   - For w/o data generator + w/o uncertainty maps make prior = false, use the data provided in the dataset section
   - For end to end ensemble make endtoend = True
   - For running with different encoders change the architecture in config
   - Also add the deatils of wandb in the configuration file.
   - In order to get the required data for training, please refer to the Dataset section. 

Then run: 
```
   cd image_dissimilarity
   python train.py --config configs/train/default_configuration.yaml
```
The following file can be run to train the model in kaggle : [Link(kaggle notebook)](https://www.kaggle.com/mlrc2021anonymous/synboost-pytorch)

### Evaluation
To Run ensemble(with grid search):
1) For FS Static: 
```
cd image_dissimilarity
python test_ensemble.py --config configs/test/fs_static_configuration.yaml 
```

2) For Lost and Found : 
```
cd image_dissimilarity
python test_ensemble.py --config configs/test/fs_lost_found_configuration.yaml 
```

To Run testing directly with fixed weights(this also saves the final prediction images where the anomaly has been segmented, for this the model path base in wandb must be provided):
1) For FS Static: 
```
cd image_dissimilarity
python test.py --config configs/test/fs_static_configuration.yaml 
``` 
2) For Lost and Found : 
```
cd image_dissimilarity
python test.py --config configs/test/fs_lost_found_configuration.yaml 
```
To run testing directly without wandb ( for this to run the path to .pth file of the model must be provided, this also saves the final predictions)
1) For FS Static:
```
cd image_dissimilarity
python test_without_wandb.py --config configs/test/fs_static_configuration.yaml --load_path <Path to .pth file>
```
2) For FS Lost and Found:
```
cd image_dissimilarity
python test_without_wandb.py --config configs/test/fs_static_configuration.yaml --load_path <Path to .pth file>
```
   
### Results

![alt text](https://github.com/manideep1108/synboost/blob/master/display_images/Comapring%20oututs%20of%20ours%20and%20authors.jpeg?raw=true)
The above image compares author's final predictions (2nd column from right) with our predictions(last column).


<p align="center">
  <img src="https://github.com/manideep1108/synboost/blob/master/display_images/main%20results.png?raw=true" alt="Sublime's custom image"/>
</p>
<p align="center">
  The above table shows the results on the the Fishyscapes private test data
</p>

<p align="center">
  <img src="https://github.com/manideep1108/synboost/blob/master/display_images/table%202.png?raw=true" alt="Sublime's custom image"/>
</p>
<p align="center">
  The above table shows the results on the Fishyscapes Validation datasets
</p>

<p align="center">
  <img src="https://github.com/manideep1108/synboost/blob/master/display_images/light.png?raw=true" alt="Sublime's custom image"/>
</p>
<p align="center">
  The above table shows the results of Light version 
</p>

<p align="center">
  <img src="https://github.com/manideep1108/synboost/blob/master/display_images/ensemble.png?raw=true" alt="Sublime's custom image"/>
</p>
<p align="center">
  The above table shows the results of end to end ensemble
</p>



### Google Colab Demo Notebook
A demo of the anomaly detection pipeline can be found here: https://colab.research.google.com/drive/1HQheunEWYHvOJhQQiWbQ9oHXCNi9Frfl?usp=sharing#scrollTo=gC-ViJmm23eM

### ONNX Conversion 

In order to convert all three models into `.onnx`, it is neccesary to update the `symbolic_opset11.py` file from the
original `torch` module installation. The reason for this is that `torch==1.4.0` does not have compatibility for `im2col`
which is neccesary for the synthesis model. 

Simply copy the `symbolic_opset11.py` from this repository and replace the one from the torch module inside your project environment. 
The file is located `/Path/To/Enviroment/lib/python3.7/site-packages/torch/onnx`



## References
[1] Learning to Predict Layout-to-image Conditional Convolutions for Semantic Image Synthesis.
Xihui Liu, Guojun Yin, Jing Shao, Xiaogang Wang and Hongsheng Li.

[2] Improving Semantic Segmentation via Video Propagation and Label Relaxation
Yi Zhu1, Karan Sapra, Fitsum A. Reda, Kevin J. Shih, Shawn Newsam, Andrew Tao, Bryan Catanzaro.

[3] https://github.com/lxtGH/Fast_Seg

[4] High-resolution image synthesis and semantic manipulation with conditional gans.
Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. 

[5] The cityscapes dataset for semantic urban scene understanding. 
Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele
