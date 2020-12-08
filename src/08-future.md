## The Future

Looking ahead, neural networks show a lot of promise to become widely used tools
in many kinds of products and systems. More than other areas of AI, the
potential for them to grow is very high. This is because, unlike other areas of
AI and machine learning, once a model has been trained it will be very simple to
take an out-of-the-box system and incorporate it into a larger service. Even
more important, the ability to take a highly-tuned model and merely swap out a
single layer and re-train it to fit a new problem (this is called a transfer
task which we'll explore in depth below) will lower the barrier to making useful
systems.

The exact path into the future will depend on progress in several complimentary
technologies (namely GPUs). Regardless, there are a number of promising
applications and emerging areas that seem guaranteed to be influential in the
coming years. Through this section we will outline a number of those and attempt
to shed some light on where the future road of neural networks will take us.


### Future of Academic Research

Serving as an extension of the current research, many types of extended
convolutional neural network models have been created in order to solve specific
problems or to extend the utility of the original model. Firstly, current
convolutional neural networks operate on images that are approximately 227 x 227
pixels.  By increasing computational and algorithmic power of these systems,
larger images can be analysed and potentially yield more accurate results.
Furthermore, sparse multi-dimensional convolutional networks have been created
which can work on arbitrary point-cloud data as opposed to a fixed image. By
working on a point-cloud, information can be learned about 3D models. This could
lead to major advances in robotic automation where a neural network powered
robots could make inferences about the space it sees from its kinect-like
sensors. In addition, there has been work on networks like SimNet which take the
lessons learned from convolutional networks (such as parameter sharing and
pooling) and extends them to new operations outside of convolutions. This will
make it possible to apply such models on a much more diverse dataset as long as
the appropriate convolution-like operation can be defined.

![With recurrent neural networks, we allow backwards connections between layers.  In this case, the hidden layer uses the input layer ​and​ the output layer to compute it's value. This encodes a time dependency which is why recurrent networks are fantastic at sequence analysis](figures/08/recurrent-01.png)

The technique with a more promising future is recurrent neural networks which
allow a layer of neurons to have connections to any other layer regardless of
its position in the network (this is the very thing that feed forward neural
networks disallow). The ability to have backwards connections gives the network
an understanding of temporal relationships which is important when doing
sequence classification such as predicting the next word in a sentence or using
multiple frames of a video to classify what is being shown.  In fact, for
robotics this sort of object classification is gaining more importance since it
produces much less noise when doing object detection with a moving camera.
Recurrent neural networks have also shown to produce very robust language models
which can be used in a wide variety of applications, from tagging sentences,
extracting entities in an article, to captioning images. In fact, the image
captioning demonstration that was shown to the press used a convolutional neural
network to feed into a recurrent neural network language model.

A beautiful aspect of recurrent neural networks is that they are Turing
complete.  This means that we can consider the trained weights of a recurrent
neural network (RNN) to represent the definitions of a program that can compute
anything that a normal computer could.  It turns the form of backpropagation
that runs on it into a pseudo-programmer that programs the RNN in order to
solve a particular task.  This may seem like a novelty, however the underlying
theory of this proves the exceptional nature of RNN's; while feed forward
networks are universal approximators and can approximate any decidable problem,
RNN's are Turing complete and can fully compute any decidable problem!

Another subfield gaining popularity are transfer task oriented networks As
discussed in <<finetuning>>, transfer tasks are when we train a model for one
problem and test it on another problem.  The exciting part about transfer tasks
becoming a higher priority in the field of deep learning is that it ushers in
our models learning to describe the world as it is as opposed to simply what the
data tells it.  That is to say, even though we are simply teaching the model to
classify a cat versus a dog, the model itself learns deeper features than the
simple task demanded from it and essentially is smarter than the problem needs.
This is particularly exciting in light of data being seen as an imperfect
representation of a slice of the world -- a model being good at a transfer task
means that it is learning to see through this limited vantage point and arrive
at understandings about the world.

We are still a long way away from having very generalizable models that work on
wide arrays of transfer tasks. However these tasks are becoming a larger part
of deep learning research and at the same time we are gaining more "neural
power" with recurrent neural networks. This makes us confident that recurrent
neural networks with large focus on transfer tasks are the most promising avenue
for the future of artificial intelligence.

Finally, very practically, there is work on making neural networks smaller and
easier to compute.  This work has multiple fronts, both on the hardware side in
making GPUs more powerful and making the actual models smaller.  One
particularly interesting attempt at making the models smaller is to use lower
precision numbers within the model (eg, storing 3.1 instead of 3.14159)
^[http://arxiv.org/abs/1502.02551].  This has the potential of
simplifying the computational overhead of neural networks and making them more
widespread on mobile and embedded devices.


### 2030: The World Deep Learning Built

On the road towards a more unified scheme for artificial intelligence, there
will also be many intermediary capabilities which will bring the benefits of
deep learning to many new fields.

There have recently been advances in turning sequences of 2D video frames into
full featured 3D models of an environment. Although this is possible without a
neural method, the current models are able to deal with noise and more exotic
scenes much more robustly than previously possible. This has been quite a boon
for robotics -- it's one thing for your roomba to be able to understand the
layout of your space, but quite another to have it understand the objects, what
they are and how to interact with them.

![Recurrent neural networks have mastered Atari games](figures/08/videogames-01.png)

These sorts of alternate uses of convolutional layers of neural networks (eg,
switching the fully connected layers from a convolutional neural network with
other types of layers) have already been around for a couple years.  For
example, using Q-Learning ^[http://mgazar.net/academic/SQLCamReady.pdf]
computers have been trained to learn how to play Atari games.
^[https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf]  This training was
done by simply letting the computer see the screen as a series of images, giving
it a method of pressing buttons on the controller and also telling it what its
current score is.  This was enough for the system to learn how to interpret the
gameplay and the relevant visual queues on screen in order to get a high score!
^[For a demo of a comparable system, see:
http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html] This sort of
learning in which a computer learns the intricacies of a system and how to
interact with it to get a desired outcome is incredibly exciting for robotics
and control systems in general!

Another great innovation we are starting to see is the combination of linguistic
models with other models. For example, recently we have been able to combine an
image model with a recurrent language model in order to automatically and
robustly caption images. ^[http://arxiv.org/abs/1411.4555] These
captions are not templates or from a database, but rather a linguistic
representation the model itself has created given an image!  This sort of
capability could be expanded for use in accessibility services for the blind or
as a general-service natural language generation method.  

On the other hand, the problem could also be inverted where language is the
input and something else is the output. For example, a user could describe an
object and have the model interpret the meaning and create the desired object.
In the year 2050, will we simply tell our home computer that we'd like a vase to
hold 12 flowers in the shape of a Klein bottle and have our in-house 3D printers
hear that, create the model and print one for us?

We could also imagine applications where image processing is advanced to a
degree where it can understand subtleties that currently require human experts.
For example, a system could be built that takes in video frames from someone's
phone and helps them diagnose problems with their cars.  As the network builds
predictions, it can be fed back to the user as a request to zoom into particular
parts of the car for further inspection.  We are already seeing this sort of
trend in expert neural network systems in the field of healthcare where data is
already in a computer-usable format.  In that field, neural networks are taking
in MRI data, along with patient data, in order to aid doctors in diagnosis.
