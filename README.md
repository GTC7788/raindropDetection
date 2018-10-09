# ON THE IMPACT OF VARYING REGION PROPOSAL STRATEGIES FOR RAINDROP DETECTION AND CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORKS


![Demonstration of Raindrop Detection](https://github.com/GTC7788/raindropDetection/blob/master/utils/ExampleofRaindropDetection.jpg)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Raindrop detection by using different region proposal algorithms

![Demonstration of Raindrop Detection](https://github.com/GTC7788/raindropDetection/blob/master/utils/InceptionModelV1.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;InceptionV1 architecture


*(This repository contains the raindrop classification, detection demonstration code and associated supporting files)*

## Abstract
The presence of raindrop induced image distortion has a significant negative impact on the performance of a wide 
range of all-weather visual sensing applications including within the increasingly important contexts of visual 
surveillance and vehicle autonomy. A key part of this problem is robust raindrop detection such that the potential 
for performance degradation in effected image regions can be identified. Here we address the problem of raindrop 
detection in colour video imagery by considering three varying region proposal approaches with secondary classification 
via a number of novel convolutional neural network architecture variants. This is verified over an extensive dataset 
with in-frame raindrop annotation to achieve maximal 0.95 detection accuracy with minimal false positives compared to 
prior work. Our approach is evaluated under a range of environmental conditions typical of all-weather automotive 
visual sensing applications.

## Reference implementation

This raindrop detection approach was based on various region proposal algorithms to propose regions within an image, and using classify 
each region by using pre-trained CNN (AlexNet, InceptionV1).

This repository contains ```raindrop_classification.py```, ```raindrop_detection_sliding_window.py``` and ```raindrop_detection_super_pixel.py``` files 
corresponding to raindrop classification based on AlexNet, raindrop detection based on sliding window and superpixel from the paper, these approaches 
demonstrate the best accuracy as shown in the paper.

To use these scripts, the pre-trained CNN network models must be [downloaded](https://www.dropbox.com/s/wp6wmn7nmli5e0f/Model.zip?dl=0&file_subpath=%2FModel).

Training datasets:
* The custom dataset used for raindrop classification can be found in [here](https://www.dropbox.com/s/nirra2nem8vopas/dataset_code_video.zip?dl=0&file_subpath=%2Fdataset%2Fraindrop+classification+dataset)
* The custom dataset used for raindrop detection can be found in [here](https://www.dropbox.com/s/nirra2nem8vopas/dataset_code_video.zip?dl=0&file_subpath=%2Fdataset%2Fraindrop+detection+dataset)

## Usage of each directory and script in this repository

**Model**

This directory should contains 4 files, due to the file size limitation of GitHub, I have put these model files in [dropbox](https://www.dropbox.com/s/wp6wmn7nmli5e0f/Model.zip?dl=0) 
to download separately.

`alexRainApr06.tfl`: trained model for AlexNet, required for raindrop detection.

`alexRaindropApr12.tfl` (3 files): trained model for AlexNet, required for raindrop classification.



**ground_truth_labels**

This directory contains 13 xml files that store the ground truth raindrop coordinates for associated images in `raindrop_detection_images`.
Those xml files will be required when highlighting the ground truth raindrops (with red rectangle) during raindrop detection for an image.



**raindrop_classification_images**

This directory contains 16 sample images can be used for raindrop classification.



**raindrop_detection_images**

This directory contains 13 sample images can be used for raindrop detection.



**raindrop_classification.py**

python script for raindrop classification.



**raindrop_detection_sliding_window.py**

python script for raindrop detection based on sliding window algorithm.



**raindrop_detection_super_pixel.py**

python script for raindrop detection based on super pixel algorithm.

## Instructions to test pre-trained models for raindrop classification and detection

1. Clone the repository.

    ```
    $ git clone https://github.com/GTC7788/raindropDetection.git
    ```

2. [Download pre-trained CNN models](https://www.dropbox.com/s/wp6wmn7nmli5e0f/Model.zip?dl=0&file_subpath=%2FModel) and put all 4 model files into the **Model** directory.

3. For **raindrop classification**, the script takes 1 argument indicating the image to process. For example:
    ```
    $ python raindrop_classification.py 3 
    ```

    The above command will process image no.3 in the _raindrop_classification_images_ folder. 

4. For **raindrop detection by using sliding window** as region proposal algorithm, the script takes 1 argument indicating the image to process. For example:
    ```
    $ python raindrop_detection_sliding_window.py 3 
    ```
    The above command will process image 3 in the in the _raindrop_detection_images_ folder and use the associated ground truth xml file in _ground_truth_labels_ folder for image 3 as well.

5. For **raindrop detection by using super pixel** as region proposal algorithm, the script takes 1 argument indicating the image to process. For example:
    ```
    $ python raindrop_detection_super_pixel.py 3 
    ```
    The above command will process image 3 in the in the _raindrop_detection_images_ folder and use the associated ground truth xml file in _ground_truth_labels_ folder for image 3 as well.



## Example video
[![Examples](https://github.com/GTC7788/raindropDetection/blob/master/utils/VideoCoverSlidingWindow.jpg)](https://youtu.be/ImF6VNtrC5Y)

Video Example for Raindrop Detection with Sliding Window - click image above to play.

[![Examples](https://github.com/GTC7788/raindropDetection/blob/master/utils/VideoCoverSuperPixel.jpg)](https://youtu.be/iuioJEi6GNE)

Video Example for Raindrop Detection with Super Pixel - click image above to play.

## Requirements

**System environment and libraries requirement**
```$xslt
1. Linux Ubuntu 16.0 or later
2. TensorFlow v1.1
3. TFLearn v0.3
4. Python3.5.2

(A installation guide for TensorFlow and TFLearn can be found at:  http://tflearn.org/installation/)
```

## Reference

[On the impact of varying region proposal strategies for raindrop detection and classification using convolutional neural networks](http://breckon.eu/toby/publications/papers/guo18raindrop.pdf)
(Guo, Akcay, Adey and Breckon), In Proc. International Conference on Image Processing IEEE, 2018.
```
@InProceedings{guo18raindrop,
  author =     {Guo, T. and Akcay, S. and Adey, P. and Breckon, T.P.},
  title =      {On the impact of varying region proposal strategies for raindrop detection and classification using convolutional neural networks},
  booktitle =  {Proc. International Conference on Image Processing},
  pages =      {1-5},
  year =       {2018},
  month =      {September},
  publisher =  {IEEE},
  keywords =   {rain detection, raindrop distortion, all-weather computer vision, automotive vision, CNN},
}

```
