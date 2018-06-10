# Microstructure Analysis of Metal in Welding Joints

Research on automating analysis of microstructures in welding joints. In collaboration with [Prof. Patricio Mendez](https://sites.ualberta.ca/~ccwj/people/professors/mendez/) and his team at the [Canadian Centre for Joining and Welding](https://sites.ualberta.ca/~ccwj/).

## Background
The goal of this project is to train a CNN to identify classes of microstructures in images of metal. The microstructures in a metal are the patterns observed under a microscope at the micrometer scale (i.e. 10-100x). The microstructure of a metal can strongly influence physical properties such as strength, toughness, ductility, hardness, corrosion resistance, high/low temperature behaviour or wear resistance of the metal. One can determine significant information about the physical properties of a metal through simply studying an image of microstructures. 

A microstructure image of a weld joint:
![microstructures](https://github.com/kevwang1/Welding-Microstructures-Classification/blob/master/Sample_Data/Segmentation_Images/Image_1.jpg)

Depending on the type of metal, the welding temperature, the cooling rate after welding, etc., different microstructures will form in the weld joint. Visual analysis of microstructures in welding joint metal is a common technique for quality assurance in welding processes. However, this is a very time-consuming task. Normally, a metallurgist will perform visual analysis by overlaying a grid of 1000 points over the image and identifying the microstructure class at each point in the grid. This point-counting method takes 30 min to 1 hour per image. Then, the statistics about the composition of microstructures is used to infer properties about the metal.

Example of point-counting microstructure classification:
![point counting](https://github.com/kevwang1/Welding-Microstructures-Classification/blob/master/Sample_Data/Point_Counting_Images/Example.png)

The class FS (Ferrite and Second Phases) can be identified by parallel lines, whereas the class AF (Acicular Ferrite) can be identified by lines in a "basket weave" pattern. The chaotic ordering in AF increases toughness and prevents crack propagation, whereas FS may crack easily along its parallel direction.

## Results

#### Point Counting Approach
Our initial approach at automating microstructure classification involving trying to emulate the point-counting technique; we divided each image into 50x50 pixel subimages, and trained a CNN to classify these image patches. Although this approach showed promise, the CNN could not make use of contextual information outside of the 50x50 pixel patch, losing the ability to make predictions that respect global structure. For example, it may place random points of other classes in a large region of Class 1, which is inaccurate. Therefore, we decided to move to a segmentation approach, which would make use of global structure in its predictions.

#### Segmentation Approach
Our images are labelled with [Labelbox](https://labelbox.com/). We are using a Fully Convolution Network for segmentation.

Sample Annotations:
- blue: PF (Primary Ferrite)
- pink: FS (Ferrite and Second Phases)
- red/yellow: AF (Acicular Ferrite)
- unlabelled: M (Martensite)

![annotation 1](https://github.com/kevwang1/Welding-Microstructures-Classification/blob/master/Sample_Data/Segmentation_Annotations/Image_1.png)
![annotation_2](https://github.com/kevwang1/Welding-Microstructures-Classification/blob/master/Sample_Data/Segmentation_Annotations/Image_2.png)

Sample Predictions:
- red: PF (Primary Ferrite)
- green: FS (Ferrite and Second Phases)
- orange: AF (Acicular Ferrite)
- brown: M (Martensite)

![prediction 1](https://github.com/kevwang1/Welding-Microstructures-Classification/blob/master/Sample_Data/Segmentation_Predictions/Image_1.png)
![prediction 2](https://github.com/kevwang1/Welding-Microstructures-Classification/blob/master/Sample_Data/Segmentation_Predictions/Image_2.png)

## Continuing Work
The results with the segmentation approach are promising. We are continuing to gather more training data and improve our training process.

For more details, please email kev.wang at ualberta.ca.
