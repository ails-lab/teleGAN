<h1 align="center">Text to Image Synthesis</h1>

<p align="justify"> This repository hosts all the code related to my diploma thesis, titled "Text to Image Synthesis Using GANs", 
for my degree in Electrical and Computer Engineering at the National Technical University of Athens. 
The main focus of this thesis is to propose a novel architecture which will be able to generate high-resolution 
images conditioned on a given text description. In addition, we examine the impact of different text representations 
on the quality of generated images from well-known models. </p>

<h2>Abstract</h2>
<p align="justify">
The problem of text-to-image synthesis is a research area that combines the fields of Computer Vision and Natural Language Processing. 
The goal is to create a model which, given a text description, generates images. These images must not only be realistic but also contain 
visual details that match the aforementioned text description. 
</p>
 
<p align="justify">
The emergence of Generative Adversarial Networks (GANs) marked a period of significant pro-gress in this direction. 
The systems that have been proposed can generate high-resolution images that match their corresponding text description 
using a variety of techniques. Stacked GANs probably constitute the most important development in this direction. 
Existing models generate an initial image of low quality, which passes through a number of sketch-refinement processing stages 
in order to generate the high-resolution image. </p>

<p align="justify">
In this diploma dissertation, we propose a novel architecture (TeleGAN) to generate high-resolution images. In particular, we use the 
Stacked GANs structure, with three stages, in order to decompose the difficult problem of generating images of high quality 
into more manageable sub-problems. More specifically, the network of the first stage generates a black and white image of 
128x128 resolution. At the second stage, colors are added to the image of the first stage. Finally, at the third and last stage, 
the image of the second stage is enhanced to high resolution (256x256). </p>

<p align="justify">
In addition, we examine the impact of different text representations, produced by char-CNN-RNN, 
GPT-2 and RoBERTa language models, on the quality of generated images from gan-int-cls and 
StackGAN models on Oxford-102 and CUB datasets. We also train these networks on the Flickr8k 
dataset and produce results.
</p>

<h2>Author</h2>
Thanos Masouris (<a href="https://github.com/ThanosM97">ThanosM97</a>)


