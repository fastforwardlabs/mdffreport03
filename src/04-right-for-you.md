## Are Neural Networks Right for You?

There are many things to consider when deciding whether to use neural networks
in your system. While very powerful, they can also
be very resource intensive to implement and to deploy. As a result, it is
important to make sure that other options have already been exhausted. Trying
simpler methods on a given problem will at the very least serve as a good
benchmark when validating the effectiveness of the neural model. However, for
images, non-neural methods are quite limited unless auxiliary data is available
(for example, quality-controlled user-generated tags with standard clustering
and classification algorithms could be a good first-pass solution).

### Picking a Good Model

If a neural model seems like the only solution, one of the most important
considerations when starting is whether a model
(particularly a trained model) that solves the problem already exists. These pretrained
models have the advantage of being ready to use immediately for testing and
already having a predefined accuracy on a given dataset, which may be good
enough to solve the problem at hand. Getting to those accuracies often
involves tuning many parameters of the model, called _hyperparameters_; this
can be quite tedious and time-consuming, and requires expert-level understanding
of how the model works in order to do correctly. Even with all other hurdles
taken care of, simply figuring out a good set of hyperparameter values for
the model can determine whether the resulting model will be usable or not.

There are many pretrained models out there to use. Caffe, for example, provides
a _model zoo_ where people can share models in a standardized way so they can be
easily used.  One that is of particular interest is the `googlenet` model, which
comes with an unrestricted license and does ImageNet classification, out of the
box, with 68.7% accuracy for predicting the correct tag and 89% accuracy for
having the correct tag in the top five results. While some of the models in the
zoo are released under a commercial-friendly license, many are distributed using
a non-commercial license (normally as a result of restrictions on the underlying
dataset). However, these pretrained models can at least serve as a good basis
for testing to see if the particular approach is valid for your problem before
going ahead and training the same model on your own custom data.

#### Notable Models in the Model Zoo
- Places-CNN: Trained on images of various locations and of various objects
    - https://github.com/BVLC/caffe/wiki/Model-Zoo#places-cnn-model-from-mit
- FCN-Xs: Segments images to find the locations of objects in images
    - https://github.com/BVLC/caffe/wiki/Model-Zoo#fully-convolutional-semantic-segmentation-models-fcn-xs
- Salient Object Subitizing: Finds the number of objects in an image
    - https://github.com/BVLC/caffe/wiki/Model-Zoo#cnn-models-for-salient-object-subitizing
- Binary Hash Codes: Generates semantic image hash codes for fast "similar image"
retrieval
    - https://github.com/BVLC/caffe/wiki/Model-Zoo#model-from-the-cvpr2015-deepvision-workshop-paper-deep-learning-of-binary-hash-codes-for-fast-image-retrieval
- Age/Gender: Predicts the age and gender of a person through an image
    - https://github.com/BVLC/caffe/wiki/Model-Zoo#models-for-age-and-gender-classification
- Car Identification: Identifies the model of a car
    - https://github.com/BVLC/caffe/wiki/Model-Zoo#googlenet_cars-on-car-model-classification
    
:::info

#### Fine Tuning / Transfer Learning

Once a pretrained model is found, it can either be used outright or run through
a process called _fine-tuning_
^[http://cs231n.github.io/transfer-learning/]. In fine-tuning, a
pretrained model is used to initialize the values for a new model that is
trained on new data. This process shows how robust neural networks can be -- a
model trained for one purpose can very easily be converted to solving another
problem. For example, a model used to classify images can be fine-tuned in order
to rank Flickr images based on their style
^[http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html].
A benefit of this is that the pretrained model already has some abilities in
recognizing images, or whatever task it was intended for, which means the
fine-tuning training is more focused and can be done with much less data. For
applications where a pretrained model that solves the given problem cannot be
found, and an adequately sized dataset is not available, it may be necessary to
find a "good enough" pretrained model and use fine-tuning to repurpose it for
the given problem. As mentioned in the description of <<convnet,convolutional
neural networks>>, often convolutional layers are reused since their ability to
extract objects from a scene are not necessarily greatly affected when changing
the downstream classification task at hand.
^[http://arxiv.org/abs/1411.1792]

:::


### Datasets and Training

One of the biggest difficulties with training your own neural network is finding a
large enough dataset that fits the problem space. While deep neural networks
can be trained to perform a wide variety of tasks, as more layers are added
(and thus the total number of parameters of the model increases), the amount
of data necessary for training also increases. As a result, when deciding
whether it is possible to train your own neural network to solve a problem,
you must consider two main questions: "Do I have enough data?" and "Is my data
clean and robust enough?"

Unfortunately, there are no easy ways to know _a priori_ whether your dataset is
large enough for the given problem. Each problem introduces its own
subtleties that the neural network must learn to figure out -- the subtler
the differences between the example data, the more examples are necessary
before the model can figure them out.

A good rule of thumb is to compare the results of your cost function between
the training and validation sets, also known as _training loss_ and _validation
loss_.  Commonly we aim at having a training loss that is a bit higher than the
validation loss when performing backpropagation. If the
training loss is about the same as the validation loss, then your model is
_underfitting_, which means you should increase the complexity of the model,
adding layers or connections. If the training loss is much lower than the
validation loss, then your model may be _overfitting_. Solutions to this include
decreasing the model's complexity or increasing the dataset size (synthetically
or otherwise).

Furthermore, when training a convolutional neural network, it is useful to look
at the actual kernels  to gauge the performance of the
network while it's being trained. We expect the kernels to be smooth and not look
noisy. The smoothness of the resulting kernels is a good measure of how well
the network has converged on a set of features. Noisy kernels could result from
noisy data, insufficient data, an overly complex model, or insufficient training
time.

One common way to synthetically increase the dataset size in an image-based task
is through multisampling, where each image is cropped in multiple ways and
flipped horizontally and vertically. Sometimes, noise is even introduced into
the input every time a piece of data is being shown to the network. This
method is recommended in every application, not only because it increases the
dataset size, but also because it makes the resulting network more robust for rotated
and zoomed-in images. Alternatively, the dropout method discussed in
Section 2 (a type of regularization) can be used. It is generally advisable to
always use the dropout method with a small dropout factor to prevent overfitting
whenever there is limited data available.

![Training your neural network](figures/03/training-graph-01.png)

However, if tweaking the model complexity, dataset size, or regularization
parameters doesn't fix the validation and training losses, then your dataset may
not be robust enough. This can happen if there are many examples of one category
but not many of another (say, 1,000 images of cats but only 5 of scissors), or
if there is a general asymmetry in the data that allows the neural network to
learn auxiliary features. A common example of this in image training is the
picture's exposure and saturation. If all pictures of cats are done using
professional photography equipment and pictures of scissors are taken on phones,
the network will simply learn to classify high-quality images as cats. This
problem shows itself quite often in the use of pretrained networks on social
media data -- many pretrained networks use stock photography as their training
set since it is highly available, however there are vast differences between the
quality and subject of stock pictures and pictures found on social media
websites. One solution is _normalizing_ images, a procedure where the mean pixel
value of the entire dataset is subtracted from each image in order to deal with
saturation and color issues. However, in cases where the dataset the model is
being applied to differs drastically from the training set, it may be necessary
to fine-tune the model, or start from scratch.

### Testing What You've Made

Finally, it is important to understand the model you have created or intend to
use. Even though neural networks are non-interpretable models (meaning we
cannot gain too much insight into the internals of how the model is making
the decisions it is), we can at least understand the domain that the model is
applicable for by looking at the training set and having a robust enough test
set.

For example, if we create a network that can estimate the amount of damage done
to a region by a natural disaster using satellite imagery, what will the model
say for regions that were completely unaffected? Is it biased to specific
architectural or geographical features because of a bias in the dataset? Or
maybe those biases simply emerged because the model was not robust enough to
extract deeper features. Having a holdout sample of the dataset that is only
used once the model is fully trained can be invaluable in understanding these
sorts of behaviors; however, a careful analysis of the dataset itself is
important.

Furthermore, consideration must be given to cases where the neural network fails to
give a high-confidence result, or simply gives the wrong result entirely.
Currently, accuracies of >85% at image processing tasks are considered cutting edge;
however, this means that for any high-volume application many incorrect results
are being given. Results from a neural network also come with confidences, so
a threshold in quality should be recognized for the given task and downstream
applications should have procedures for when no results match the given
confidence level. Another route is to use the results of the neural network to
inform further algorithms in a way that can potentially increase the confidence,
or at least the usability, of the results. In our prototypes, we use a
hierarchical clustering on the predicted labels in order to increase the
usability of low-confidence results. This draws on
the intuition that even if an image cannot be confidently classified as a cat,
most of the labels with nonzero confidences will be under the WordNet label
"animal," and this is a sufficiently informative label to use in this case.

### Timelines

Below are some suggested timelines for working with neural networks in
different situations. It is important to note that these numbers are
_incredibly_ approximate and depend very much on the problem that is being
solved. We describe a problem as a "solved problem" if there already exists a
pretrained model that does the specified task.

| Situation                       | Research | Data Acquisition | Training | Testing
| ---                             | ---      | ---              | ---      | ---
| Pre-Trained Model Available     | 1 week   | days             | N/A      | 1 week
| Similar to Pre-Trained Model    | 1 week   | 1 week           | days     | 1 week
| Similar to Problem in a Paper   | weeks    | 1 week           | 1 week   | 1 week
| New application of known model  | weeks    | 1 week           | weeks    | 1 week
| New application and novel model | months   | weeks            | months   | weeks


### Deploying

Deploying a neural network is relatively easy; however, it can be quite
expensive. The trained model is large -- easily 500 MB for a single moderately
sized network. This means that `git` is no longer an adequate tool for
versioning and packaging the datafile, and other means should be used. In the
past, using filename-versioned models on Amazon's S3 has worked quite well,
particularly when done with the S3 backend for `git-annex`.

The machine that the model gets deployed on should have a GPU and be properly
configured to run mathematical operations on it. This can be difficult initially
to set up; however, once installed correctly, backups of the machine can easily
be swapped in and out ^[To see the process in AWS, check out
http://tleyden.github.io/blog/2014/10/25/cuda-6-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/].
The main complication comes from installing `cuda` if you are using an NVIDIA
device, as well as installing `cuda`-enabled math libraries such as `theano`,
`caffe`, `cudnn`, `cufft`, etc.

Once the model file is on a machine properly configured to use its GPU, using
the neural model is quite the same as using any other model. A common route
for facilitating a distributed cluster of these models is to wrap the network
itself in a thin lightweight HTTP API and deploy it to a cluster of 
GPU-enabled machines. Then, any service in your ecosystem that must take
advantage of the model's power can pick a server in this cluster using a
round-robin approach -- new models can be uploaded and, one by one, computers
in the neural network cluster can be updated without disrupting downstream
services.

![Recommended architecture for your neural network service](figures/03/architecture-01.png)

Having a policy for rolling out new models is quite important. Models
generally need to be updated as their usage changes and new/different datasets
become available that could increase their accuracy. It is very much suggested
to instrument any service that uses the results of the neural network in order
to obtain feedback data for use in future training (for example, asking the
user "Was our result good?" or "What better response could we have provided?").

### Hardware

As described in _2 Neural Networks_, while we think of these models as a series
of neurons connected to each other with various activation functions, the
results are actually computed with a series of vector operations. In fact, most
neural networks can be seen as simply taking the linear combination of vectors,
applying some nonlinear function (a popular choice is the `tanh` function), and
maybe taking a Fourier transform. These computations are perfectly suited for a
GPU, which has been optimized at the hardware level to perform linear algebra at
high speeds.

This is why NVIDIA has been very strongly pushing for using its GPUs for
general mathematics as opposed to simply gaming. They have even gone as far
as creating very specialized math libraries that optimize these linear
algebra operations on their devices and, in recent months, developing specialized neural network hardware for their next-generation,
computation-specific GPUs.

These considerations have gone from being useful for the academic working on
these problems to necessary for anyone working with neural networks -- our
models are getting more and more complex, and the CPU is no longer adequate for
training or evaluating them. As a result, infrastructure using neural models
_must_ have GPUs in order to function at acceptable speeds. When working on
Pictograph, we found that we could perform a single image classification in
about 6 seconds on the CPU, vs. 300 ms on the GPU (using the `g2.2xlarge` AWS
instance). Furthermore, the operations scale very well on a GPU (often
incurring almost no overhead if multiple images are classified together).
