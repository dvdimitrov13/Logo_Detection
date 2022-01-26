<!-- PROJECT SHIELDS -->

[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/dvdimitrov13/Logo_Detection">
    <img src="images/logo.png" alt="Logo" width="200" height="200">
  </a>

  <h1 align="center">Logo Detection</h1>

  <p align="center">
   Knowing who is using your product and in what context is power! By using Computer Vision we can acquire organic customer insights by detecting the right images on social media. This repository showcases models trained to detect 15 distinct brand logos in pictures/videos online!
  </p>
</div>

<!-- TABLE OF CONTENTS -->
## Table of Contents
<details>
  <ol>
    <li>
      <a href="#about-the-project">About the project</a>
      <ul>
        <li><a href="#environment">Environment</a></li>
        <li><a href="#development">Development</a></li>
        <li><a href="#results">Results</a></li>
      </ul>
    </li>
    <li>
      <a href="#user-manual">User manual</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#training">Training</a></li>
        <li><a href="#detection-and-evaluation">Detection and evaluation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About the project

This project was developed as part of my Data Science degree coursework, the goal was to maximize the accuracy. After a brief experimentation with ResNet50, [YOLOv5](https://github.com/ultralytics/yolov5/blob/master/README.md) was identified as the best performing model for the job. Additionally, thanks to it's state of the art architecture I was able to develop two distinct models one can choose to implement:
* yolo1000_s_cust - based on YOLOv5_s architecture 
* yolo1000_x - based on YOLOv5_x architecture

The different architectures allow the user a tradeoff between speed and accuracy. Here is a more detailed comparison between the two architectures (statistics based on [pretrained checkpoints](https://github.com/ultralytics/yolov5/blob/master/README.md)):

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>CPU b1<br>(ms) |Speed<br><sup>V100 b1<br>(ms) |Speed<br><sup>V100 b32<br>(ms) |params<br><sup>(M) |FLOPs<br><sup>@640 (B)
|---                    |---  |---    |---    |---    |---    |---    |---    |---
|YOLOv5s      			|640  |37.2   |56.0   |98     |6.4    |0.9    |7.2    |16.5
|YOLOv5x      			|640  |50.7   |68.9   |766    |12.1   |4.8    |86.7   |205.7

<p align="right">(<a href="#top">back to top</a>)</p>


<a name="env"></a>

### Environment
The main model development enironment was an Azure instance with an NVIDIA K80 with 12 GB of vram. Additionally, I used Google Collab with GPU acceleration enabled in order to test different hyperparameter configurations. 

Finally, in order to make the reproducebility and improvement of this repository as straightforward as possible I used Git Large File Storage, allowing for a simple cloning that includes all relevant training data as well as model weights.

<div align="center" style="position:relative;">
    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-colab-small.png" width="60" height="60"/>
    </a>
    </a>
    <a href="https://git-lfs.github.com/">
        <img src="https://github.com/dvdimitrov13/Logo_Detection/blob/master/images/git_lfs.png" width="60" height="60"/>
    </a>
    <a href="https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwiS0f6Bt7j0AhWRzXcKHXXkA0gYABAAGgJlZg&ae=2&ohost=www.google.com&cid=CAESQOD2WXDDC3bcaN6__E7gY08J137qyTW6nOQb8DRsJPfVaCbKW_MnwwecmS8dCR7oZPQSLYd6V8LfB32ZLnpJUqA&sig=AOD64_2JGNrArPWvbnOJLMOXwqSsXl1gSw&q&adurl&ved=2ahUKEwjrovWBt7j0AhW3gv0HHXPGCYYQ0Qx6BAgCEAE&dct=1">
        <img src="https://aspiracloud.com/wp-content/uploads/2019/07/azure.png" width="60" height="60"/>
    </a>
    <a href="https://pytorch.org">
        <img src="https://user-images.githubusercontent.com/74457464/143897946-feabe8ec-ac91-4d38-a8e0-f1db1db3d84b.png" width="25%" height="60"/>
    </a>
 
</div>

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
### Development

#### 1. Dataset
The raw dataset provided 17 Brand Logos, after initial inspection Intimissimi and Ralph Lauren were dropped due to large amount of mislabeled data. This left me with 15 Logos:  Adidas, Apple Inc., Chanel, Coca-Cola, Emirates, Hard Rock Cafe, Mercedes-Benz, NFL, Nike, Pepsi, Puma, Starbucks, The North Face, Toyota, Under Armour.

In order to convert the dataset to YOLOv5 PyTorch TXT Format I used  [Roboflow](https://roboflow.com/). As far as preprocessing I applied auto-orient and image resize (to the correct model size 640x640). Since YOLOv5 already applies data augmention in its training script which can be finutened through hyperparameters, no data augmentation steps were taken in Roboflow.

After applying additional preprocessing steps descibed in the following section, the final dataset (can be found under the name [yolo1000](https://github.com/dvdimitrov13/Logo_Detection/tree/master/formatted_data/yolo1000)  has the follwoing statistics:

 <div align="center">
    <a href="https://roboflow.com/?ref=ultralytics">
        <img src="https://github.com/dvdimitrov13/Logo_Detection/blob/master/images/descriptive_stats.png" />
    </a>
</div>

#### 2. Training 
For model training transfer learning was used. Instead of starting from randomized set of weights I used the pretrined checkpoints provided by YOLOv5 repo, which are already trained on the COCO dataset.

To establish baseline performance I first trained a model using an unaltered version of the full dataset (around 38K images) using YOLOv5l. Following that I took the following steps to improve performance:
* Improved label consistency by adding missing bounding boxes, original data had only one box per image
* Improved label accuracy by resizing bounding boxes with inaccurate position 
* Balanced the class distribution by including a maximum of 1000 images per class
* Added 2% background images (218 images) and removed wrong bounding boxes in order to reduce False Positives

By applying those the not only did the performance of the much smaller YOLOv5s model reach the baseline performance but also there was a significant improvement with instances of false positves, example below.

YOLOv5l - baseline         |  YOLOv5s - v1          |
:-------------------------:|:-------------------------:|
![](https://github.com/dvdimitrov13/Logo_Detection/blob/master/images/yolo_full_l.jpg)  |  ![](https://github.com/dvdimitrov13/Logo_Detection/blob/master/images/yolov5s.jpg)  |

Finally, through experimentation with hyperparameters the YOLOv5s - v1 was further improved, the following data augmentation steps were applied: HSV -Hue/Saturation/Value, Translation, Scale, Flip (horizontal), Shear, Mosaic and Mixup. 

Both final models - YOLOv5s - v2 and YOLOv5x - final, were trained using those finetuned hyperparameters.

### Results 
For model evaluation the main metric used was mAP. Additionally, the IoU (Intersection over union) was calculated at  bounding box confidence of 50% in order to exclude "lucky guesses".

 Here the average results for each model: 
  
| Model| mAP<sup>val<br>0.5 |mAP<sup>val<br>0.5:0.95  
| :-----: | :-: | :-: 
| YOLOv5l - baseline | 0.870 | 0.701
| YOLOv5s - v1 | 0.868 | 0.693 
| YOLOv5s - v2 | 0.873 | 0.711
| YOLOv5x - final | 0.881 | 0.722

We can see that the performance of the two final models is very close, while the speed of the X model is significantly lower, therefore if the user wants absolute optimal performance over a set of images **YOLOv5x - final** shoudl be used, while **YOLOv5s - v2** is better suited for realtime video detection.

Here are the final results for **YOLOv5x - final**:
| Logo| mAP<sup>val<br>0.5 |mAP<sup>val<br>0.5:0.95 | IoU<br>>50% confidence 
| :-----: | :-: | :-: | :-: 
| Adidas 		| 0.837 | 0.713 | 0.924 
| Apple Inc. 	| 0.936 | 0.764 | 0.928 
| Chanel 		| 0.830 | 0.644 | 0.844 
| Coca Cola 	| 0.889 | 0.648 | 0.867 
| Emirates	 	| 0.786 | 0.688 | 0.919
| Hard Rock Cafè | 0.953 | 0.813 | 0.930 
| Mercedes Benz | 0.908 | 0.778 | 0.920 
| NFL 			| 0.921 | 0.744 | 0.892 
| Nike 			| 0.781 | 0.630 | 0.908 
| Pepsi 		| 0.721 | 0.530 | 0.815 
| Puma 			| 0.869 | 0.679 | 0.887 
| Starbucks 	| 0.964 | 0.869 | 0.940 
| The North Face | 0.958 | 0.818 | 0.923 
| Toyota 		| 0.907 | 0.749 | 0.874 
| Under Armour 	| 0.963 | 0.771 | 0.909 

## User manual

### Installation
### Training
### Detection and evaluation


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/dvdimitrov13/Logo_Detection/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/dimitarvalentindimitrov/
[product-screenshot]: images/screenshot.png
