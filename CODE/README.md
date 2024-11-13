# SAETNMDA
MiRNA-disease association prediction based on Stack Auto-Encoders and Triplet networks

## Requirements
  * python==3.8.18
  * numpy==1.24.3
  * scikit-learn==1.3.0
  * pandas==2.0.3
  * tensorflow==2.4.0
  * xgboost==2.0.2
  * xlrd==1.2.0
  * openpyxl==3.1.2

## File
### data
  All data input files needed to run the model are in the folder ```../IN```, which contain folders:
  * ```ORI DATA```: contains original input data which are used for HMDDv2.0 and HMDDv3.2 in the same name folders, respectively.
    - miRNA functional similarity matrix.txt: MiRNA functional similarity
    - disease semantic similarity matrix 1.txt and disease semantic similarity matrix 2.txt: Two kinds of disease semantic similarity
    - known disease-miRNA association number.txt: Validated mirNA-disease associations
    - disease number.txt: Disease id and name
    - miRNA number.txt: MiRNA id and name
  * ```Q18.3``` and ```Q18.3_HMDD3``` contain proceeded data files for HMDDv2.0 and HMDDv3.2, respectively.
    - Folder ```Q18.3/kfold/``` contains data input files respective for folds in 5-fold-CV on HMDDv2.0.
    - Folder ```Q18.3_HMDD3/kfold/``` contains data input files respective for folds in 5-fold-CV on HMDDv3.2.
    - Folder ```Q18.3_HMDD3/mi_k/``` contains data input files respective for De novo miRNAs on HMDDv3.2.
    - Folder ```Q18.3_HMDD3/dis_k/``` contains data input files respective for Case studies on HMDDv3.2.
      Each folder includes proceeded files:
      + SR_FS*: Integrated miRNA similarity matrix
      + SD_SS*: Integrated disease similarity matrix
      + y_train*: Training human MDAs matrix
      + y_4loai*: Human MDAs matrix which are ussed for train and test phase.
  * ```INDE_TEST``` contain proceeded data files for Independent Test.
### result
  The predictive scores in 5-fold-CV and Case studies are files ```*prob_trbX*``` in the folder ```Q23_TripletNetwork/OUT Q_*/*/Q18.3/Results/Combination/```.
### code
  * For 5-fold-CV in HMDD2 or HMDD3 and Case studies in HMDD3:
    - params.py: For changing parameters
    - model.py: Structure of the model
    - train_TripletNet1_*.py, train_TripletNet2_*.py: Train triplet networks
    - kethop_kfold_va_dis_k_2_loai.py: Train the final model.
  * For All in one in Independent test: Inde_Test.py.
  * For All in one in De novo miRNAs: Deno_mi.py.
## Usage
  * Download code and data then unzip ```IN*.rar``` to ```IN```.
  * Because of the big size of dataset, data in github is uploaded for using one repeat time running. You can edit code to run in one repeat time. For further data, please feel free send email to npxquynh@hueuni.edu.vn.
  * If ```OUT*``` directory does not exist, unzip ```OUT_EG.rar```.
  * How to run:
     - For 5-fold-CV in HMDD2 or HMDD3:
      1. Choose dataset and type of evaluation. Default: ```HMDD3``` and ```kfold```. If you want to change parameters, edit in the file ```params.py```. 
        1.1. Choose dataset:
          ```db = 'HMDD2'``` or ```db = 'HMDD3'```
        1.2. Choose type of evaluation:
          ``` type_eval = 'kfold' ```
      2. Run for 5-fold-CV:
        2.1. Run ```train_TripletNet1_kfold.py```
        2.2. Run ```train_TripletNet2_kfold.py```
        2.3. Run ```kethop_kfold_va_dis_k_2_loai.py```.
    - For Case studies in HMDD3:
      1. Choose dataset and type of evaluation.
        1.1. Choose dataset:
          ```db = 'HMDD3'```
        1.2. Choose type of evaluation:
          ``` type_eval = 'dis_k' ```
      2. Run for Case studies:
        2.1. Run ```train_TripletNet1_dis_k.py```
        2.2. Run ```train_TripletNet2_dis_k.py```
        2.3. Run ```kethop_kfold_va_dis_k_2_loai.py```.
    - For All in one in Independent test or De novo miRNAs: Run ```Inde_Test.py``` or ```Deno_mi.py```.
