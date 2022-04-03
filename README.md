# Stylegan Color and Gender Control
In  this repo we have wroked on official pytorch repo <a href="https://github.com/NVlabs/stylegan2-ada-pytorch.git" target="_blank">stylegan2-ada-pytorch</a> to control gender and color simultanoeusly for generating faces.
# Use
-Clone git hub repo in your drive
-Download the <a href="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl" target="_blank">Pre-Trained</a> pickle file from drive. and place the file inside repo folder.
-Open gender_color.py and select the required Olor seeds for selecting ethinicity. The reult will be with two faces with opposite genders
#Row Seeds for selecting Ethiniciy:

458-White with Blue eyes

75-Black (African)

85-white with brown hair

100-Asian (South Asian)

1500-White with dark complaxion

For example, if you need to select the person with  black ethinicity you need to set 'row_seeds' to '[75]' i.e. 'row_seeds=[75]'. In col_seeds we take random values for genrating faces.

-The code is optimized to run on 'CPU devices'.If you don't have 'GPU', you can run this in your machine 
