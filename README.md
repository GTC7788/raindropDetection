# Raindrop Detection

https://www.dropbox.com/s/z9b7jh5xebsddlp/Example%20of%20Raindrop%20Detection.png?dl=0

Detect raindrops on vehicle windscreen by combining various region proposal algorithm with Convolutional Neural Network.

**ground_truth_labels**

This directory contains 13 xml files that stores the ground truth raindrop coordinates for associated images in `raindrop_detection_images`.
Those xml files will be required when highlighting the ground truth raindrops (with red rectangle) when performing raindrop detection for an image.



**Model**

This directory should contains 4 files, due to the file size limitation of GitHub, I have put these model files in dropbox. 
`alexRainApr06.tfl`: trained model for AlexNet, required for raindrop detection.
`alexRaindropApr12.tfl` (3 files): trained model for AlexNet, required for raindrop classification.



**raindrop_classification_images**

This directory contains 16 sample images prepared for raindrop classification.



**raindrop_detection_images**

This directory contains 13 sample images prepared for raindrop detection.



**raindrop_classification.py**

python script for raindrop classification.



**raindrop_detection_sliding_window.py**

python script for raindrop detection based on sliding window algorithm.



**raindrop_detection_super_pixel.py**

python script for raindrop detection based on super pixel algorithm.


**System environment and libraries requirement**
```$xslt
1. Linux Ubuntu 16.0 or later
2. TensorFlow v1.1
3. TFLearn v0.3
4. Python3.5.2

(A installation guide for TensorFlow and TFLearn can be found at:  http://tflearn.org/installation/)
```

