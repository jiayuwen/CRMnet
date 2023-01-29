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


- To download the original data and preprocess the data from scratch:
    1. Download the original data from: https://zenodo.org/record/4436477#.Y4a_PS0RoUE
    2. Copy "complex_media_training_data_Glu.txt" to "./Yeast_Original_Data/"
    3. Preprocessed the data:

            python3 data_preprocessing.py
            
    4. The final dataset in tf.data.Dataset format is about 250GB

- To download the preprocessed data:
    1. Download the preprocessed data from: https://zenodo.org/record/7375243#.Y9W0iS0RoUG
    2. Load the preprocessed data by using pickle:
    
            pad_seq_list = pickle.load(open(PATH+"seq_list", "rb"))
            exp_list = pickle.load(open(PATH+"exp_list", "rb"))

- To use the trained model:
    1. Download the model weight from: https://zenodo.org/record/7375243#.Y9W0iS0RoUG
    2. Load the trained model:
    
            model = tf.keras.models.load_model(PATH)

- To train model from scratch: 
    1. on tpu v2-8 about 4 hours to converge:
    
            python3 train.py
    
       trained model will be saved on folder *"/saved_model"*

