# CelebAI
CelebAI is an ML model that can recognize the face of over 100 celebrities with an accuracy of 80% within the matter of seconds. It is trained with on a dataset containing over 80,000 images using a complex convolutional neural network (CNN). 

The model works in the following way. First, the user is directed to a home page where they can upload a celebrity's image. Once, the user does this, we have a model trained on over 80,000 training images which first crops out the face from each image using CV2's face cascade classifier. This cropped dataset is stored in a separate folder for training. We use a complex neural network to train the model and a validation split to verify our results. The obtained accuracy is over 90% for training and over 80% for testing. 

![HomePage](https://github.com/kaushikcodes/CelebAI/blob/main/pt_1.png)

For the web application, the model is stored in a .h5 file. As soon as the user uploads the image, the model almost instantaneously predicts and assigns a class to the image. The web application then redirects the user to a different page where they can view the prediction along with the initially uploaded image.

![Prediction](https://github.com/kaushikcodes/CelebAI/blob/main/pt_2.png)


[YouTube demo link](https://youtu.be/i7LqLEvLGWA) 
