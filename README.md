# SAR-Assessing-Metrics-for-Deep-Learning-Despeckling

This repository contains the pytorch implementation of four SAR assessing metrics presented in the [IEEE GRSL Paper entitled Enhanced Deep Learning SAR Despeckling Networks based on SAR Assessing metrics
](to be defined).

if you find it useful and use it for you research, please cite as the following 
```
@ARTICLE{11028079,
  author={Vitale, Sergio and Ferraioli, Giampaolo and Pascazio, Vito and Deniz, Luis Gomez},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Enhanced Deep Learning {SAR} Despeckling Networks Based on SAR Assessing Metrics}, 
  year={2025},
  volume={22},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2025.3577907}}
```


In this work, a framework for assessing deep learning based methodologies for SAR image despeckling is presented.
The definition of a DL method is based on the observation of training and validation loss and, typically, euclidean metrics, such as L1 or L2 are considered.
For going beyond the DL perspective and provide a more phisical interpretation of the training process, this framework defines a SAR based validation stage by using SAR assessing metrics in the design and hyper-parameter selection of neural networks.
In a first phase, SAR assessing metrics may be used only as validation metrics with the aim of highlighting critical issues that can not be spotted with standard image-processing
quality metrics. In a second phase, the same SAR assessing metrics may be used directly for enhancing the DL solution by the addressing specific issues arisen during the previous SAR based
validation stage.
The four SAR assessing metrics are the ENL, MoI, MoR, VoR. All of them are no-reference metrics and can be used for the validation stage of both supervised and unsupervised methods. 

![immagine](https://github.com/user-attachments/assets/7636b5c4-1c4b-4b22-a671-167020d91b2d)

![immagine](https://github.com/user-attachments/assets/904e9fa7-6454-4421-af38-66058568da77)

# Usage
import the desired metric from *sar_validation_loss.py* and follow description within the descript

all the metrics are implemented considered intensity format of the data

## Variable Defintion:
all the varianile are in intensity format and considered as tensor of four dimensions [batch_size, 1, rows, cols]

**inputs**: SAR image used as input of the DL method

**outputs**: noise-free estimate provided as output of the DL method

**ref**:	(only for supervised methods) noise-free reference


# Team members
 Sergio Vitale    (contact person, sergio.vitale@uniparthenope.it);
 
 Giampaolo Ferraioli (giampaolo.ferraioli@uniparthenope.it);
 
 Vito Pascazio (vito.pascazio@uniparthenope.it);
 
 Luis Gomez Deniz (luis.gomez@ulpgc.es)
 
# License
Copyright (c) 2025 Dipartimento di Ingegneria and Dipartimento di Scienze e Tecnologie of Università degli Studi di Napoli "Parthenope".

All rights reserved. This work should only be used for nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document LICENSE.txt
(included in this directory)


# Prerequisites
This code is written on Ubuntu system for Python3.7 and uses Pytorch library.
  
