## Neural Networks

Neural networks have been around for many years, though their popularity has
surged recently due to the increasing availability of data and cheap computing
power. The perceptron, which currently underpins all neural network
architectures, was developed in the 1950s. Convolutional neural networks,
the architecture that makes neural image processing useful, were introduced in
1980. However, recently new techniques have been conceived that allow the
_training_ of these networks to be possible in a reasonable amount of time and
with a reasonable amount of data. These auxiliary algorithmic improvements, in
addition to computational improvements with GPUs, are why these methods are
only now gaining popularity. Moreover, these methods have proven themselves to
excel at extracting meaning from complex datasets in order to properly
classify data we didn't think could be algorithmically classified before.

In this section we'll look at the basic concepts needed to understand how neural
networks function and the recent advances that have greatly expanded their
possible applications.

### The Perceptron

The basic elements of a modern neural network -- neurons, weighted connections,
and biases -- were all present in the first neural network, [the "perceptron,"
invented by Frank Rosenblatt at Cornell Aeronautical Labs in 1958](http://page.mi.fu-berlin.de/rojas/neural/chapter/K3.pdf). The design was
grounded in the theory laid out in Donald Hebb's _The Organization of Behavior_,
which gave a method for quantifying the connectivity of groups of neurons
through _weights_. The initial technology was an array of photodiodes making up
an artificial retina that sent its "visual" signal to a single layer of
interconnected computing units with modifiable weights. These weights were
summed to determine which neurons fired, thus establishing an output signal.

![The Mark I Perceptron, the progenitor of modern neural networks. Image courtesy of Cornell University News Service records, #4-3-15. Division of Rare and Manuscript Collections, Cornell University Library.](figures/02/perceptron-photo.jpg)

Rosenblatt told the _New York Times_ that this system would be the beginning of
computers that could walk, talk, see, write, reproduce themselves, and be
conscious of existence. This field has never lacked imagination! Soon
researchers realized that these statements were exaggerated given the state of
the current technology -- it became clear [that perceptrons alone were not
powerful enough for this sort of computing](http://sss.sagepub.com/content/26/3/611). However, this did mark a
paradigm shift in AI research where models would be trained (non-symbolic AI)
instead of working based on a set of preprogrammed heuristics (the Von Neumann
architecture).

As the simplest versions of neural networks, understanding how perceptrons
operate will provide us insight into the more complex systems popular today. The
features of modern networks can be viewed as solutions to the original
limitations of perceptrons.

![The basic elements of a perceptron.](figures/02/basic-perceptron-01.png)

To understand neural networks, we must first unpack the basic terminology:
individual computational units (neurons) are connected (i.e., pass information),
such that each connection has a _weight_ and each neuron a _bias_. A number that
is passed to a neuron via a connection is multiplied by the weight of that
particular connection, summed together with the other inbound connections, and
adjusted by the neuron's bias. This result is passed to an activation function
that determines whether the neuron "fires" or not. An active, or fired, neuron
passes the result on. If the result does not meet the activation threshold, then
it is not passed on.

Perceptrons are simple binary classifiers that use the above computational
components, take in a vector input (see <<inputvector>>), and output a 0 or a
1 to indicate the classification. This classification is regulated by a set of
weights learned during the training phase.

The term _neuron_ stems from the biological motivation behind neural
nets. A primary property of a brain is its ability to wire (and rewire) neurons
together so that, given some input signal (e.g., sound in your ear), groups of
neurons will fire together and activate different brain regions, leading to a
nervous system or other response. Neurons inside the brain receive input voltages
from many connections, but only fire if the current is strong enough to pass
across the synapse and carry the electrical signal to them. Similarly, the
weights in neural networks allow us to bias certain input connections more than
others to extract the relevant features.

::: info

#### Input Vectors

Like most machine learning models, neural networks require an _input vector_ to
process. An input vector is a way of quantifying an input as a series of
numbers. Neural networks operate by passing this input through layers of neurons
that transform the input vector into your desired output.

![The petal and sepal measurements of an Iris as input vector.](figures/02/iris-01.png)

If we wanted to quantify the properties of a flower as an input vector, we could
form a list of numbers describing the flower's height, the length of the petals,
three values for the color (one for each of the red/green/blue values), etc.
footnote:[This exact example is part of a classic "hello world" dataset for
machine learning called the Iris Dataset.] To quantify words the
_bag of words_ approach is generally used, where we create a "master" list in which every
possible word has a position (e.g., "hello" could be the 5th word, "goodbye" could
be the 29,536th word). Any given passage of text can be quantified using this
approach by simply having a list of 0s and 1s, where a 1 represents that that word
is present in the passage. An image, on the other hand, is already a
quantification of a visual scene -- computer image formats are simply 2D lists
of pixels, which are just numbers representing the RGB values. However, when
creating a vector out of them, we must discard the 2D nature of the data and turn
it into a flat list, thus losing any spatial relationships between the pixels.

:::

What makes a perceptron interesting is how it handles weights. To evaluate a
perceptron, we multiply each element of the input with a series of weights and then
sum them; if the value is above a certain activation threshold, the output of
the perceptron is "on." If the value is below the threshold, the output is
"off":

<figure>
<img src="figures/eq_perceptron.png" style="max-width: 260px;" />
</figure>

In this formulation, `w` encodes the weights being used in the calculation and
is a vector with the same size as the input, `x`. There is also a bias (also
called a threshold), which is simply a constant number. The result of the
function `f(x)` defines the classification. That is to say, if we train our
system such that `0` means dog and `1` means cat, then `f(a)=0` means that the
data in the vector `a` represents a dog and `f(b)=1` means that `b` represents a
cat.

While training perceptrons is much simpler than the training regimes we will
soon get into, they still do need their weights tuned in order to be useful. For
this, we must define a _cost function_, which essentially defines how far
we are from our desired output. We will go into this in more detail soon; for
now, it's useful to know that we are attempting to converge on a result that
minimizes our cost function by slowly changing our weights.

The weights in a perceptron describe a linear function that separates the input
parameter space into two sections describing the two possible classifications of
the system. As a result, only linearly separable problems can be solved. What we
mean by separability is that our parameter space (all features encoded in our
input vector) has the capability of having a line drawn through it, which at
those values creates a boundary between classes of things. This quickly limits
the effectiveness of perceptrons when applied to more complicated classification
problems.


![Perceptrons are only effective in the case of linearly separable problems.](figures/02/separability-01.png)

=== Feed-Forward Networks: Perceptrons for Real Data ===

As a result of the single layer peceptron being limited to linearly separable
problems, researchers soon realized that it could only solve toy problems in its
original formulation.  What followed were a series of innovations that
transformed the perceptron into a model that is still to this day the bread and
butter of neural networks: the feed-forward network. This involved the
modification of most of the original components, while retaining the underlying
theory of Hebbian learning that originally motivated the perceptron's design.

The feed-forward neural network is the simplest -- but most widely used -- type of
neural network. The model assumes a set of neurons with an arbitrary threshold value and
connections to the _next_ set of neurons. The first set of neurons perform a weighted
summation of their input; then, that signal is _fed forward_ to the next layer. As
connections are unidirectional toward the next layer, the resulting network has
no potential cycles, which simplifies the training procedure.

To understand how these networks work, we'll first need to amend a few of our basic concepts from the perceptron.

==== Nonlinear Activation and Multiple Layers ====

_Nonlinear activation functions_ change several things from the perceptron
model. First, the output of a neuron is no longer only 0 or 1, but any value
from 0 to 1. This is achieved by replacing the piece-wise function that defines
`f(x)` in the perceptron model with:

<figure>
<img src="figures/eq_nonlin.png" style="max-width: 260px;" />
</figure>

where `σ` is the chosen nonlinear function.^[Common
choices for this function are `tanh`, Softmax (i.e., generalized logistic
regression), or ReLU.]

![Non-linear activation increases the type of problems neural networks can be applied to.](figures/02/activation-01.png)

Next, stacking perceptrons allows for hidden layers, at the computational cost of
many more weights. Here, every node in the new layer gets its value by
evaluating `f(x)` with its own set of weights. As a result, the connection
between a layer of `N` nodes and `M` nodes requires `M` weight vectors of size
`N`, which can be represented as a matrix of size `N x M`.

![A multilevel perceptron made by stacking many classic perceptrons.](figures/02/multiple-layers-01.png)

Having multiple layers opened up the possibility of multiclass classification -- i.e.,
classifying more than two items (the limit of the perceptron). The final layer
can contain multiple nodes, one for each class we want to classify. The first
node, for example, can represent "dog," the second "cat," the third "bird," etc.; the
values these nodes take represent the confidence in the classification.

Of course, this introduces the complexity of selecting the correct classification.
Should we simply accept the class with the highest confidence, a maximum
likelihood approach? Or should we adopt another method (Bayes), where the set of
probabilities across all possible classes informs a more sophisticated choice.
Maximum likelihood is generally accepted in practice however, in our prototypes
we explore alternate methods to make low-confidence classifications useful.

Crucial to these advancements is that they allow classifications of datasets
that are not linearly separable. One way to think about this is that the hidden
layers perform transformations on the space to form linearly separable results.
But the reality is slightly more complicated. In fact, with the addition of
nonlinearity, a feed-forward neural network can act as a universal approximator.
That is to say, nonlinearity enables a neural network to model _any_ function,
with the accuracy proportional to the number of neurons. Adding multiple layers
makes it easier to attain high-accuracy models and reduces the total number of
required nodes.

Now it begins to come together how adding more layers actually escalates our
power to model, classify, and predict. However, why is it that neural networks
are so much better then traditional regression and hierarchical modeling? We've
mentioned before that we _train_ our models; now, let's take a look at how this is done.

#### Backpropagation

While these models seem fantastic, it was generally possible to train comparable
models on small datasets using classic regression techniques. The
real breakthrough for neural networks was in the learning or training procedure:
_backpropagation_. This piece of the puzzle is the reason why neural
networks outmuscled previous methods.

![Backpropagation adjusts weights and biases to better match target results.](figures/02/backprop-01.png)

Backpropagation is an optimization technique for models running on labeled data
(also known as _supervised learning_). While this algorithm had been known for
quite a long time, it [was only first applied to neural networks in 1986](http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html). In
this technique, data is fed through a randomly initialized network to identify
where the network gets things wrong.  This error is then "backpropagated"
through the network, making subtle changes to gently nudge the weights toward
better values. The goal of this training is to craft our weights and biases to
transform our input vector, layer by layer, into a separable space (not
necessarily _linearly_ separable) where it can be classified.  This is done with
successive use of the _chain rule_, which can be thought of as iteratively
seeing how much a given weight contributed to a particular result and using the
calculated error to correct the problem.

Think of a musician who is playing electric guitar on a new amp for the first
time. Her goal will be to make the tonality clear and the distortion appropriate
for the song or style, even though the amp's settings are initially random. To do
this, she'll play a chord with her sound quality goal in mind and then start
fiddling with each knob on the amp: gain, mid, bass, etc. By seeing how each
knob relates to the sound and repeatedly playing the chord, adjusting, and
deciding how much closer she has gotten to her goal, she will do a kind of
training. Listening to the chord is like evaluating the objective function,
and tuning the knobs is like minimizing the cost function.

While the exact description of the algorithm is outside the scope of this report,^[For a good in-depth treatment of backpropagation, check out
[http://neuralnetworksanddeeplearning.com/chap2.html](http://neuralnetworksanddeeplearning.com/chap2.html).] there are several
considerations to keep in mind. First, the error that is propagated through the
network is based on the cost function (also known as the loss function).
The cost function defines "how wrong" the neural network was in a given
prediction. For example, if a network was supposed to predict "cat" for a given
image but instead says it thinks 60% it was a cat and 40% that it was a dog, the
loss function would determine how much to penalize the network for the imperfect
output. This is then used to teach the network to perform better in the future.
There are many possible choices for a loss function, and each one penalizes the
network differently based on how incorrect it was versus the correct
output.footnote:[Common loss functions are categorical cross entropy, mean
squared error, mean absolute error, and hinged.] As we'll see
in <<regularization>>, other terms can also be added to the loss function to
account for other properties we wish to control (for example, we could add a
term to regulate the magnitude of the weights in the network).

<figure>
<img src="figures/eq_nonlin.png" style="max-width: 260px;" />
<figcaption>Example of a possible cost function (mean squared error).</figcaption>
</figure>

Second, backpropagation is an iterative algorithm and has a parameter, the
_learning rate_, that determines how slowly it should change the weights. It is
always advised to start with a small learning rate (generally `.001` is used:
if a change of `5.0` would precisely fix the error in the network, a
change of `.005` is applied). A small learning rate is crucial to avoid
overfitting the model to the training data because it limits the memorization of
particular inputs. Without this, our networks would learn only features specific
to the training set and would not learn generalization. By limiting the amount the
network can learn from any particular piece of data, we increase the ability of
the network to generalize. This is such an important piece of neural networks
that we even go as far as modifying the cost function and truncating the network
to help with this generalization.

Furthermore, there is no clear time when iterations are _done_, which can cause
many problems. To address this, we diagnose the network using cross-validation.
Specifically, the dataset should be split into two parts, the training set and
the validation set (generally the training set is 80% and the validation set is 20%
of the data). We use the validation set to calculate an error (also called the
_loss_), which is compared to the error on the training set. Comparing these two
values gives a good idea of how well the model will work on data in the wild,
since it never learned directly from this validation data. Furthermore, we can
use the amount of error in the training versus the validation sets to diagnose
whether training is complete, whether we need more data, or whether the model is
too complex or too simple (see <<overunderfitting>> for how to diagnose these
problems).

That backpropagation is an iterative algorithm that starts with essentially
random weights can also lead to suboptimal network results. We may have terrible
luck and initialize our network close to a local minimum (i.e., backpropagation
may yield a solution that is _better_ than our initial guess of weights, but
nowhere close to a globally "best" solution). To correct this, most researchers
train many networks with the same hyperparameters but with different randomly
initialized weights, selecting the model with the best result.

#### Regularization

Even with our current tools for building neural networks, we still face the
basic threat that all machine learning entails: overfitting. That is, even with
a complicated architecture, we risk making a neural network that only
understands our current dataset. To make sure we don't teach our pony one trick,
we need to create a robust regularization scheme.

Regularization is a set of methods to ensure the model better generalizes from
the dataset used to train it. This may not seem as glamorous as finding new
neural architectures, but it is just as important since it allows us to train
simpler models to learn more complicated classifications without overfitting.
Regularization can be applied to a wide range of models, allowing them to
perform better and be more robust.^[For an in depth treatment of
regularization see [http://neuralnetworksanddeeplearning.com/chap3.html](http://neuralnetworksanddeeplearning.com/chap3.html)]

The first form of regularization that was used is L1 and L2 regularization,
collectively known as _weight decay_ (see <<weightdecay>>). With weight decay,
we not only train our network to have the correct weights in order to solve the
problem, but we also try to nudge the weights to be as small as possible. This
is done by adding an extra term to the loss function that penalizes the model
when weights grow to be large. The intuition is that by forcing weights to be
small, we don't have any one particular weight dominating the signal.  This is
good because we want as much cooperation between nodes as possible to account
for all features when making a decision. Furthermore, when weights are large, it
is harder for our optimization procedure to drastically affect the result.
For example, if we had the weights `[0.4, 0.6, 0.2]`, we could easily affect
the output vector from that layer using backpropagation; however, the
weights `[0.4, 256.0, 0.2]` would be almost unaffected by a similarly
backpropagated error. Small weights create a simpler, more powerful model.

<figure>
<img src="figures/eq_nonlin.png" style="max-width: 260px;" />
<figcaption>Weight decay used to regulate weight growth.</figcaption>
</figure>

![Regularization helps prevent overfitting.](figures/02/regularization-01.png)

A more recent, and very popular, [form of regularization called _dropout_](http://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) is widely
used. In this method, a random subset of neurons "drop out" of a forward and
backward pass during training. That is to say, with a dropout parameter of
`0.2`, during training only 80% of neurons are ever used for forward and
backward propagations (and the weight values are scaled accordingly to account
for the reduced number of neurons). As every neuron learns aspects of the
features necessary to do the classification, this means the decision-making
process is spread more evenly across nodes. Think of this as noise being added
to the input, making overfitting rarer since the network never sees the same
_exact_ input twice.

#### Putting It All Together

In the _feed-forward model_, we transform the input vector by sending it
through neurons that use _weights_ and _biases_ to compute intermediate values.
Then we pass these values through a _nonlinear activation function_ to see if the
information moves forward. We use _multiple layers_ to allow for more general
feature extraction, since each layer gives our model complexity. Finally, we use
the result from our output layer to calculate how wrong we were with a _cost
function_. _Backpropagation_ tells us how to adjust each neuron to improve our
result, and we use _regularization_ and _dropout_ to generalize our results and
prevent overfitting.

This may seem complex, but this background is sufficient to understand,
evaluate, and engineer Deep Learning systems.

The feed-forward net is to neural networks what the margarita is to pizza: the
foundation for further exploration. Most systems you'll have encountered in the
wild prior to recent innovations were feed-forward neural networks: systems for
things like character recognition, stock market prediction, and fingerprint
recognition.  Having covered the basics, we can now start to take you from
simple systems to complex, emergent ones.

### Convolutional Neural Networks: Feed-Forward Nets for Images

![Image transformed into an input vector.](figures/02/imagevector-01-01.png)

If Deep Learning ended with feed-forward neural networks, we would have trouble
classifying images robustly. So far, our inputs have all been vectors; but
images are spatial, intrinsically 2D structures (3D if we include color). What
is needed is a neural network that can maintain this spatial structure and
still be trained with backpropagation. Luckily, this is exactly how
convolutional neural networks work ^[For a good treatment on
convolutional neural networks, check out
[http://colah.github.io/posts/2014-07-Conv-Nets-Modular/](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/)].

These networks are quite new.  They were first explored in 1980, and
gained wide spread adoption in 1998 in the form of the [`LeNet`](http://deeplearning.net/tutorial/lenet.html) ([pioneered by Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)) with their
ability to do hand-written digit recognition. However, in 2003 they were
[generalized and simplified](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.91.1367&rep=rep1&type=pdf)
into a form which allowed them to solve much more complex problems.

As the name states, instead of operating on the matrix multiplication between
the input and a set of weights, a convolutional neural network works on a
convolution ^[For a nice interactive aid in understanding convolutions,
check out [http://setosa.io/ev/image-kernels/](http://setosa.io/ev/image-kernels/)] between the input and a kernel
(see below). The kernel is simply a small matrix (generally between 3
x 3 and 8 x 8) that extracts certain spatial features from the image. This sort
of technique is used all the time in classical computer image processing. For
example, the Sobel operator that is often used for edge detection in images
works by convolving a specific set of 3 x 3 matrices with the image. With a
convolutional net, we aim to learn the values for these kernels instead of for
the full sets of weights.

![Convolutional neural networks learn to create kernels that encode the spatial features of a 2D image into a 1D feature vector. This example shows a premade kernel for a sharpen filter.](figures/02/kernel-01.png)

The full structure of a convolutional neural network is a stacking of several
convolutional layers and then several layers of a classic feed-forward neural
network. The convolutional layers can be thought of as prepping the data so
that the feed-forward layers can take advantage of the spatial structure of the
input image. This structure highlights the flexibility of neural networks in
general -- we can choose to have the convolutional layers feed into a feed-forward
neural network or any other type of neural network, depending on what the
problem demands. (In _8 The Future_ we talk about alternate setups used to solve
different types of problems, such as captioning images or training a computer to
play video games.)

When defining a layer of a convolutional neural network, we specify the number
of kernels, how big each kernel is, and the "stride" (step size, or number of
spaces moved between each kernel evaluation). This layer will output a new
"image" that has a different dimensionality from the input, with spatial
features extracted. For example, if our input image was 227 x 227 x 3 (i.e., 227
x 227 pixels over 3 colors) and we used 96 kernels of size 11 x 11 with a stride
of 4, the output of this layer would have the dimensions 55 x 55 x 96. These, in
fact, are the [exact layer parameters for the ImageNet model](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). In this formulation, each of the 55 x 55 layers is considered a _depth slice_.

![Example of the 96 11 x 11 kernels from an ImageNet convolutional neural network. Image from [http://cs231n.github.io/convolutional-networks/](http://cs231n.github.io/convolutional-networks/).](figures/activationmap.jpeg)

It is important to note that while there are incredibly high numbers of inputs
and outputs, there are only 96 x 11 x 11 x 3 weights across the entire layer.
With the addition of the 96 biases, this is a total of 34,944 parameters --
substantially fewer than the 44,892,219,387 parameters we would have had in a
normal feed-forward neural network! This is why convolutional neural networks
ushered in neural image processing. It is amazing how much processing can be
done with a convolutional neural network given the relatively small number of
parameters. The convolutional example above uses the same number of parameters
as two feed-forward layers of 186 neurons each, which is quite small for any
problem of real interest!

A method known as [_max pooling_](http://people.idsia.ch/~ciresan/data/icsipa2011.pdf), which combines
the values of pixels close to each other, further reduces the number of
parameters in convolutional networks.  This can happen on the input or for any
depth slice. Max pooling defines a region, typically 2 x 2- or 3 x 3-pixel
blocks, and only takes the maximum value in any block of that size.  This is
similar to downsampling or resizing an image: it both reduces the dimensionality
of the data and adds a form of translational invariance that safeguards the
neural network when the scene is shifted in one direction or another (i.e., if
most pictures of cats have the cat in the center of the image, but we still want
it to perform well if the cat is on the side of the image). We can see the
result of these dimensional reductions in <<convnet-example>>: note how the
resolution of the depth slices is reduced at every layer.

![A simple convolutional neural networks processing an input image.](figures/02/convnet-01.png)

In practice, convolutional networks are used as a feature extractor for classic
feed-forward networks. A convolutional neural network takes an image as input
and processes it through many layers of convolutions. Once the image has been
treated through enough layers, the output of the final convolutional layer is
reshaped into a vector and fed into what is called a "fully connected" layer.
This may seem like exactly what we were avoiding -- once we reshape the data
into a vector, we lose the spatial relationships we were trying so hard to
maintain. The intuition, however, is that after the image has been passed
through multiple convolution steps, the neurons will have been encoded with all
the relevant spatial features. For example, if the image had a diagonal edge,
there would be some neurons will have encoded that pattern, and therefore
rendering the actual spatial data at that point is redundant.

Once in the fully connected layer, we can simply classify as before. In fact,
the fully connected region of the network is where we actually start making the
associations between the spatial features seen in the image and the actual
classifications that we require the network to make. One might think of the
convolutional steps as simply learning to look at the picture the right way
(i.e., looking for specific color contrasts, edges, shapes, etc.), while the
fully connected layers correlate the appearance of particular features with
classification classes. In fact, the spatial features seen by the convolutional
layers are often robust enough that they can be kept static when training a
network on multiple tasks, while only the weights for the fully connected layers
are changed from problem to problem in a method called fine-tuning. 

Fine-tuning has raised the potential of neural networks as common components
within a service. It allows us to use them in _transfer tasks_: when we take
part of a pretrained model and [port it over to a different task by fine-tuning
the new layers](http://arxiv.org/abs/1411.1792). For convolutional
networks, this means we can continue to improve models that encode spatial
features while utilizing them in diverse classification problems. Task transfer
allows us to iterate on specific pieces of our architecture while modularly
building systems for new problems. In many cases, this saves time and
computational cost because parts of a robust neural network can be connected
into new layers trained around a particular problem.

### What Is Deep?

Having taken the long walk with us through the building blocks of this
technology, you may still be wondering, "What exactly is this Deep Learning
thing?" Since you're likely to hear this term trumpeted by many new companies
and the media in the near future, we want to make sure it's given some context
-- or should we say, depth.

As previously shown, layers in a neural network can be stacked as desired, at the
cost of computation and the requirement for more data. However, with each new layer, the neural network is able to create more robust
internal representations of the data. This can allow a deep neural network to
tease out very subtle features from the data to accurately classify inputs that
would cause other models to fail.

It is important to realize, however, that when we speak of "deep" learning, we
are not simply referring to the number of layers. While there is no concrete
definition of what "deep" means in this context, it is generally accepted that
the number of causal connections each neuron has is a more accurate
representation of the depth. That is to say, if a particular neuron's output can
affect a large number of other neurons through many possible paths, that network
is considered deep. ^[See Section 3 in
[http://arxiv.org/pdf/1404.7828.pdf](http://arxiv.org/pdf/1404.7828.pdf)] This focus on the number of possible causal
connections allows us to think of powerful but small architectures as deep. For
example, simple two-layer recurrent neural networks may not have many layers,
but the robustness of the neural connections creates many causal links,
resulting in a network that can be seen as very deep.

