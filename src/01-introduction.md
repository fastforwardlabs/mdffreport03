## Introduction

![Image object recognition could allow computers to guide us through car repair, plant care, and bug bite triage.](figures/01/example-circles-01.png)

Imagine fixing your car by taking a picture of your engine and having an AI
mechanic guide you through the repairs. Or a machine that looks at a rash or bug
bite and tells you whether it needs professional attention. Or maybe a program
that looks at your garden and warns you which plants are at risk of dying.
These ideas may sound like science fiction, but they are now becoming feasible.
Recently, we've seen massive progress in the development of systems that can
automatically identify the objects in images using a technique known as Deep
Learning. This is a breakthrough capability.

![Greater theoretical understanding, affordable GPUs, and accessible datasets are motivating large advances in image object recognition.](figures/01/three-advances-01.png)

These systems are emerging _now_, due to multiple factors. First, there's been
strong progress in our theoretical understanding of artificial neural networks.
Neural networks are computational systems made up of individual, interconnected
processing nodes that adapt to new input. While they've been around since the
1950s, this recent progress has opened up entirely new applications.

Second, graphical processing unit (GPU) computation has become affordable. GPUs
were primarily developed for video gaming and similar applications, but are also
optimized for exactly the kinds of operations that neural networks require.

Finally, large image and video datasets are now available. This, more than
anything, has motivated and enabled significant progress in both research and
industry applications. The result is that we are now able to build affordable
systems that can analyze rich media (images, audio, and video) and automatically
classify them with high accuracy rates.

This has strong implications for anyone building data processing and analytics
systems. Current approaches are, out of necessity, largely limited to analysis
of text data. This limitation (frequently glossed over by many analytics
products) comes from the fact that images can be permuted in many more ways than
sentences. Consider that the English language only contains roughly 1,022,000
words, yet each pixel from an image can take on any of 16,777,216 unique color
values. Moreover, a single 1024 x 768-pixel image contains as many pixels as
Shakespeare had words in all of his plays!

Neural networks, however, are fantastic at dealing with large amounts of complex
data because of their ability to internally simplify and generalize their inputs.
Already, with high accuracy, we are able train machines to identify common
objects in images. It's exciting to think that we are now able to apply the same
kinds of analyses that we've been doing on text data to data of all types.

The structure of neural networks was initially inspired by the behavior of
neurons in our brains. While the brain analogy is a romantic one, the
relationship between these systems and the human brain stops there. These
machines do a very good job of solving very specific problems but are not yet
able to approach generalized intelligence. We don't need to worry about the
Terminator just yet.

In this report we explore Deep Learning, the latest development in multilayered
neural networks. These methods automatically learn abstract
representations of their training data to perform some sort of classification or
regression task, where you are training a system to look at examples of data
with labels and apply those labels to new data. While Deep Learning has
implications for many applications, we focus specifically on image analysis
because this is one domain where similar results cannot currently be achieved
using more traditional machine learning techniques. Deep Learning represents a
substantial advancement in image object recognition.

![Image recognition is used by ATMs to identify check amounts.](figures/01/check-01.png)

Neural network-based image recognition systems have actually been used in the wild
for quite some time. One of the first examples of neural networks applied in a product
is [the system that recognizes the handwriting on checks deposited into ATMs](http://yann.lecun.com/ex/research/), automatically figuring out how
much money to add into any account.

Image analysis is just the beginning for Deep Learning. In the next few years,
we expect to see not only apps that can look at a photo of leaky plumbing or a
damaged car and guide you through the repairs, but also apps that offer features
such as [realtime language translation in videoconferencing](http://googleresearch.blogspot.ie/2015/07/how-google-translate-squeezes-deep.html) and even machines that can diagnose diseases accurately.
