# udacity_computerVision_imageCaptioning
Image Captioning Project for Udacity Computer Vision Course 


# Project Goal
This project’s goal is to create an Image Captioning model, that applies a short text caption to an input photo.  
Models like these have several uses.  One use is to make an image database text searchable, by applying several captions to each database image and then searching the database by caption text.

# Approach
To achieve image text captioning an Encoder – Decoder model is used.  The model’s initial Encoder segment uses a CNN to extract features from each image.  These features are stored in a vector and represent the objects of interest present in the image.  This feature vector is then passed to the model’s Decoder segment.
The model’s Decoder segment applies a text caption to an input image.  The Decoder segment accepts an image feature vector as input.  This feature vector is fed into an NLP Embedding Layer to organize it in a spatially meaningful way??.  Then the embedded feature vector is fed into several LSTM layers.  These LSTM layers look at the feature vector and decide what text caption to apply image.  A single sentence caption is generated one word at a time and this sentence caption is returned as the model’s output.
Exploring Different Model Designs and Hyperparameter Choices
In an effort to the find the best Image Captioning Model, several different model designs and hyperparameter choices were explored.  Adding an attention mechanism to the model design proved productive.  This attention mechanism allows the model to focus on specific image features, from the CNN model Encoder Segment, when generating a text caption.  (For more details, please see the Attention Mechanism section.)  
Different hyperparameter choices were explored during model selection.  These hyperparameter choices included:
LSTM Depth – number of stacked LSTM layer.  A good depth settings is needed for good model performance and reasonable model training times.
Optimizer Choice – ADAM optimizer is the preferred model trainer.
Batch Size – How many images to train on simultaneously.
Vocab Threshold – Number of vocab words to remember from training corpus.
Embedding Size – Length of dense embedding vector.
Hidden Size – Size of hidden LSTM state vector passed between LSTM layers.
Number Epochs – Number of training loops to complete before stopping the training process.

Many different model designs and hyperparameter choices were explored.  Each candidate choice was trained and validated on the same training data subset.  Training each design choice on a subset of the data allows for faster iterations and controls cost.  The best design was then trained on the entire training set and presented as the best image captioning model.

# AWS Setup
AWS’s Sagemaker service was used to train the model.  The training data was downloaded from the MS COCO website, on the AWS Sagemaker Domain Space as an EFS instance.  Then the model was trained by running the following Jupyter Notebook Scripts.  
	1_Preliminaries.ipynb
	2_Training.ipynb
	3_Inference.ipynb

# Creating a Sub-Sample Training and Validation Dataset to Control Cost
  Sub-sampling technique here.

# Results
	Results here.

# Repo Organization
  Each design iteration has its own branch to allow the developer to quickly switch between different design choices.

# What is Attention
A good Attention Mechanism focuses the LSTM layers on specific image features when that LSTM network creates the output Text Caption.  The embedded feature vector from the CNN Encoding layer, and a context vector are fed into the Attention Mechanism.  The Attention Mechanism then uses linear algebra and tanh activation layer to decide, for each word in the output caption, what image feature to focus on.  The Attention Mechanism is used for each word in the output sentence, and it can focus on different image features for different words in the output sentence.
Benefits of an Attention Mechanism are better image text captions and faster training times.

# What is an Embedding Layer
An Embedding Layer is a NLP device that maps each word into a text caption into a dense vector of a defined size.  Each word gets becomes its own dense vector that has mostly non-zero elements.  
The Embedding Layer maps words into a NLP vector space in a special way.  After the mapping, the word associations translate into a spatial organization that depends on distance and spatial correlation.  This dense vector encoding is very space efficient too (compared to one hot-encoding).
Surprisingly, the Embedding Layer’s mapping weights are learned during the training process.  These weights begin as a random numbers, with a Glorot uniform initialization.  It is kind of magic that the ADAM optimizer can find a good embedding transformation!
