# ON THE IMPACT OF VARYING REGION PROPOSAL STRATEGIES FOR RAINDROP DETECTION AND CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORKS

![Demonstration of Raindrop Detection](https://github.com/GTC7788/raindropDetection/blob/master/utils/Example%20of%20Raindrop%20Detection.png)

*This project contains the raindrop classification and detection demonstration code (and associated supporting files).*

## Usage of directories:

**ground_truth_labels**

This directory contains 13 xml files that store the ground truth raindrop coordinates for associated images in `raindrop_detection_images`.
Those xml files will be required when highlighting the ground truth raindrops (with red rectangle) during raindrop detection for an image.



**Model**

This directory should contains 4 files, due to the file size limitation of GitHub, I have put these model files in [dropbox](https://www.dropbox.com/s/wp6wmn7nmli5e0f/Model.zip?dl=0) 
to download separately.

`alexRainApr06.tfl`: trained model for AlexNet, required for raindrop detection.

`alexRaindropApr12.tfl` (3 files): trained model for AlexNet, required for raindrop classification.



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


## Requirements:

**System environment and libraries requirement**
```$xslt
1. Linux Ubuntu 16.0 or later
2. TensorFlow v1.1
3. TFLearn v0.3
4. Python3.5.2

(A installation guide for TensorFlow and TFLearn can be found at:  http://tflearn.org/installation/)
```

