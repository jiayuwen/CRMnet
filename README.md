# CRMnet: a deep learning model for predicting gene expression from large regulatory sequence datasets


    .
    ├── preprocessed_data/                          # Preprocessed data folder
    ├── train.py                                    # Model training code
    ├── model.py                                    # Model construction code        
    ├── requirements.txt                            # required python packages
    └── README.md



This repository contains code for "CRMnet: a deep learning model for predicting gene expression from large regulatory sequence datasets"

- To setup environment on TPU vm (take v2-8 as example):
    
    1. initiate a tpu-vm with tensorflow 2.8.0 preinstalled:
     
            gcloud alpha compute tpus tpu-vm create tpu_v2 --zone=asia-east1-c --accelerator-type=v2-8 --version=tpu-vm-tf-2.8.0

    2. install supporting packages:

            pip install -r requirements.txt

- To download preprocessed data in tf.data.Dataset format (~250Gb):
    1. The download link will updated soon
    2. Copy the data to "./preprocessed_data/"

- To download the orignal data and preprocess the data from scratch:
    1. Download the orignal data from: https://zenodo.org/record/4436477#.Y4a_PS0RoUE
    2. Copy "complex_media_training_data_Glu.txt" to "./Yeast_Original_Data/"
    3. Preprocessed the data:

            python3 data_preprocessing.py

- To train model from scratch: 
    1. on tpu v2-8 about 4 hours to converge:
    
            python3 train.py
    
       trained model will be saved on folder *"/saved_model"*

