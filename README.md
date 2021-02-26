# Deep-Learning-and-Art
# Goal
Generating novel artistic images using the Neural Style Transfer algorithm created by [Gatys et al. (2015).](https://arxiv.org/abs/1508.06576)

Neural Style Transfer (NST) is one of the most fun techniques in deep learning. As seen below, it merges two images, namely: a **"content" image (C) and a "style" image (S), to create a "generated" image (G**). 

The generated image G combines the "content" of the image C with the "style" of image S. 

In this example, an image of the Louvre museum in Paris (content image C) is mixed with a painting by Claude Monet, a leader of the impressionist movement (style image S).
<img src="images/louvre_generated.png" style="width:750px;height:200px;">

Here are few other examples:

- The beautiful ruins of the ancient city of Persepolis (Iran) with the style of Van Gogh (The Starry Night)
<img src="images/perspolis_vangogh.png" style="width:750px;height:300px;">

- The tomb of Cyrus the great in Pasargadae with the style of a Ceramic Kashi from Ispahan.
<img src="images/pasargad_kashi.png" style="width:750px;height:300px;">

- A scientific study of a turbulent fluid with the style of a abstract blue fluid painting.
<img src="images/circle_abstract.png" style="width:750px;height:300px;">

# Model Overview
A [VGG-19](https://arxiv.org/abs/1508.06576) CNN  is used to perform Transfer Learning. The idea of using a network trained on a different task and applying it to a new task is called transfer learning.

Neural Style Transfer (NST) algorithm is done in three steps:

- Build the content cost function J_{content}(C,G)
- Build the style cost function J_{style}(S,G)
- Put it together to get J(G) = alpha * J_{content}(C,G) + beta * J_{style}(S,G) where 'C' is the content image, 'S' is the style image and 'G' is the generated image. Alpha and Beta are hyperparameters that control the relative weighting between content and style. 

# References:

The Neural Style Transfer algorithm was due to Gatys et al. (2015). Harish Narayanan and Github user "log0" also have highly readable write-ups from which we drew inspiration. The pre-trained network used in this implementation is a VGG network, which is due to Simonyan and Zisserman (2015). Pre-trained weights were from the work of the MathConvNet team. 

- Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) 
- Harish Narayanan, [Convolutional neural networks for artistic style transfer.](https://harishnarayanan.org/writing/artistic-style-transfer/)
- Log0, [TensorFlow Implementation of "A Neural Algorithm of Artistic Style".](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)
- Karen Simonyan and Andrew Zisserman (2015). [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf)
- [MatConvNet.](http://www.vlfeat.org/matconvnet/pretrained/)
- Convolutional Neural Networks, Deep learning Specialization by Deeplearning.AI : https://www.coursera.org/learn/convolutional-neural-networks
