
# Image Colorization Starter Code

The objective is to produce color images given grayscale input image.

## Solution

- A brief report on various methods I tried is attached as report.pdf

##  Running this code

-  All the requiremets are written in requirements.txt

- To train,  python main.py [args - we can find all the options at the start of main.py]

- Use predict.py to convert a set of gray scale images into color images (paste images in preds/input/all and colorized images will be in preds/output). So change the paths in pred.py accordingly. Comments added in pred.py

- vgg_loss.py contains vgg feature loss implementation 

- colorize_data.py contains dataloader

- models folder contains 2 models i used

- runs folder contains tensorboard runs and saved models (best and last) and runs out images at each iteration

- split.py to split train test data into a folder of dataloader format.

- uitls.py conatin all the util functions  

- All the trained models are saved in the runs folder and also tensorbaprd metrics at each epoch are also saved under runs folder.
