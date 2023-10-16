# Object_Detection
Here the project is Object Detection Using OpenCV Python and COCO Names & MobileNet SSD
**What is an Mobilenet SSD?**
Mobilenet SSD is an object detection model that computes the output bounding box and object class from the input image. This Single Shot Detector (SSD) object detection model uses Mobilenet as a backbone and can achieve fast object detection optimized for mobile devices.
**What exactly is coco.names? Heard a lot about it.**
Always remember that for a deep learning or machine learning model to have a good precision, a large dataset is a necessity. So, here’s where our file coco.names comes into picture. COCO stands for Common Objects in Context. This dataset contains objects from an everyday context. It is easier to identify a bottle in front of a blank wall than when it is in a classroom where children are playing hopscotch in the background. COCO dataset provides the labeling and segmentation of the objects in the images. There are 80 object categories of labeled and segmented images in the file. Thus our model can detect and identify 80 types of objects.

**Requirement**
-->COCO Names
-->Mobilenet (ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
--> frozen_inference_graph.pb

The above models are latest 2020 to detect the objects


![image](https://github.com/varmabharath30/Object_Detection/assets/83111946/8e978088-3763-4cde-9346-b8d3ce485797)


Combining MobileNets and Single Shot Detectors for fast, efficient deep-learning based object detection
If we combine both the MobileNet architecture and the Single Shot Detector (SSD) framework, we arrive at a fast, efficient deep learning-based method to object detection.

The MobileNet SSD was first trained on the COCO dataset (Common Objects in Context) and was then fine-tuned on PASCAL VOC reaching 72.7% mAP (mean average precision).
**Here the final Output**

![image](https://github.com/varmabharath30/Object_Detection/assets/83111946/97fc9651-473e-4c53-ab0d-56436752d920)



