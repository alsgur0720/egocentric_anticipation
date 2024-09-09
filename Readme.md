# Disentangled Adaptive Fusion Transformer using Cooperative Adversarial Perturbation for Egocentric Action Anticipation


## Environment

- The code is developed with CUDA 12.2, ***Python >= 3.10.0***, ***PyTorch >= 2.0.0***

    0. [Optional but recommended] create a new conda environment.
        ```
        conda create -n ego-a3 python=3.10.0
        ```
        And activate the environment.
        ```
        conda activate ego-a3
        ```

    1. Install the requirements
        ```
        pip install -r requirements.txt
        ```


### Data Structure

   * EK100 dataset:
       ```
          $YOUR_PATH_TO_EK_DATASET
          ├── rgb_kinetics_bninception/
          |   ├── P01_01.npy (of size L x 1024)
          │   ├── ...
          ├── flow_kinetics_bninception/
          |   ├── P01_01.npy (of size L x 1024)
          |   ├── ...
          ├── target_perframe/
          |   ├── P01_01.npy (of size L x 3807)
          |   ├── ...
          ├── noun_perframe/
          |   ├── P01_01.npy (of size L x 301)
          |   ├── ...
          ├── verb_perframe/
          |   ├── P01_01.npy (of size L x 98)
          |   ├── ...
       ```


   * EGTEA Gaze+ dataset:
       ```
          $YOUR_PATH_TO_EGTEA_DATASET
          ├── rgb_kinetics_bninception/
          |   ├── OP01-R01-PastaSalad.npy (of size L x 1024)
          │   ├── ...
          ├── target_perframe/
          |   ├── OP01-R01-PastaSalad.npy (of size L x 107)
          |   ├── ...
          ├── noun_perframe/
          |   ├── OP01-R01-PastaSalad.npy (of size L x 54)
          |   ├── ...
          ├── verb_perframe/
          |   ├── OP01-R01-PastaSalad.npy (of size L x 20)
          |   ├── ...
       ```



## Training

The commands are as follows.

```
cd Ego_A3
# Training from scratch
python tools/train_net.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES



# Inference

cd Ego_A3
python tools/test_net_rev.py --config_file $PATH_TO_CONFIG_FILE --gpu $CUDA_VISIBLE_DEVICES \ MODEL.CHECKPOINT $PATH_TO_CHECKPOINT MODEL.CHECKPOINT $PATH_TO_REV_CHECKPOINT



    
