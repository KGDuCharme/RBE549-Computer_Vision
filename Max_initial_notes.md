# General Notes:

## Expected Goal:
Given a camera with a top-down view camera, be able to identify an assortment of objects (probably tools, other things you’d want to know where it is on a desk), and track their locations, in pseudo real-time. Will likely use deep learning of some sort.

## Targets:  
* Be able to train on own data.
* Track objects locations.
* Demoable, so live tracking. (Maybe harder than we know, will see)

### Potential Reach Targets (unlikely, but cool):
* Gesture recognition.
* Know where objects usually are, and alert if they aren’t there for a prolonged period of time.
* Semantic labeling (eg “The pen is on the left side of the table”).

## Considerations:
* Training conditions.
        * What camera to train on.@Kevin, if you have a raspberry pi I think the RPi camera would be a good, not expensive option. In that case, training would likely be off device, or it just acts as a camera interface. Alternatively, can find cheap webcam, or only have one person train and collect the data.
        * Lighting conditions. Should be fairly consistent, as I think it would be consistent in real applications as well. Maybe a light with the camera.
        * ML/CV library. Will most likely be in python, whether we go with OpenCV or, like, tensorflow/pytorch, depends on how easy it is to do things. 
        
# Object Tracking Methods:
* Usually a combo of a detector and a tracker.
## Detection:
* In the vehicle tracking example that I had done at NVIDIA GTC, the following methods were used:
        * Single shot multi-box detector with mobile nets.
        * Faster RCNN with inception resNet.
* Faster RCNN
        * Pros: Accuracy
        * Cons: Slow, hard to implement, hard to train. 
* SSD
        * Pros: Easier to work with.
        * Cons: accuracy.
* On top of that, there’s the actual network infrastructure.
        * mobileNets
                * A platform recently developed by Google, useful for resource-constrained situations.
        * ResNet 
        
## Tracking:
* Tracking is following an object once it has been detected.
* Usually much Faster than detection.
* Common method is to detect the object every N frames, and track the object for the N-1 frames in between detections.
* TLDR, Probably want to use KCF
* OpenCV has many diff options in 3.2
        * Boosting, Multi Instance Learning (MIL), Kernelized Correlation Filters (KCF), Medianflow, TLD.
        * Boosting is an old algorithm, MIL is ok, but KCF works faster and more accurate than MIL, TLD has a lot of false positives, Medianflow is robust, requires offline training.
* The basic concept is, once given a known box, from frame to frame, move the box to the spot nearby that is closest to the positive box (in a l2-norm type context).   