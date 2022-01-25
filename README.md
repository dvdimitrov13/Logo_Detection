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
      <a href="#getting-started">Review and improve</a>
      <ul>
        <li><a href="#prerequisites">Installation</a></li>
        <li><a href="#installation">Training</a></li>
        <li><a href="#installation">Evaluation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
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
        <img src="https://user-images.githubusercontent.com/74457464/143897946-feabe8ec-ac91-4d38-a8e0-f1db1db3d84b.png" width="20%" height="45"/>
    </a>
 
</div>

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- DEVELOPMENT -->
### Development

#### 1. Dataset
The raw dataset provided 17 Brand Logos, after initial inspection Intimissimi and Ralph Lauren were dropped due to large amount of mislabeled data. This left me with 15 Logos:  Adidas, Apple Inc., Chanel, Coca-Cola, Emirates, Hard Rock Cafe, Mercedes-Benz, NFL, Nike, Pepsi, Puma, Starbucks, The North Face, Toyota, Under Armour.

In order to convert the dataset to YOLOv5 PyTorch TXT Format I used  [Roboflow](https://roboflow.com/). As far as preprocessing I applied auto-orient and image resize (to the correct model size 640x640). Through testing I found that data augmentation in Roboflow hurt model accuracy since the YOLOv5 training script already implements data augmentation which can be finutuned using hyperparameters.

Finally, I found that a balanced dataset improved training accuracy, therefore I opted to sample a maximum of 1000 iages per class and further reduce the data to 10000 images for easier annotation. The final dataset named [yolo1000](https://github.com/dvdimitrov13/Logo_Detection/tree/master/formatted_data/yolo1000) has the following statistics:

 <div align="center">
    <a href="https://roboflow.com/?ref=ultralytics">
        <img src="https://github.com/dvdimitrov13/Logo_Detection/blob/master/images/descriptive_stats.png" />
    </a>
</div>
 
### Results

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



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

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/dvdimitrov13/Logo_Detection/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/dimitarvalentindimitrov/
[product-screenshot]: images/screenshot.png
