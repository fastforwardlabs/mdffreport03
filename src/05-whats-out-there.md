## Deep Learning in Industry Today

Neural networks have been deployed in the wild for years, but new progress in
Deep Learning has enabled a new generation of products at startups, established
companies, and in the open source community.

In the startup community we've seen several companies emerge with the aim of
making Deep Learning affordable and approachable for any product manager or
engineer, as well as companies that apply Deep Learning to a specific problem
domain, such as medicine or security. There's been similar growth in the open
source community, as companies and academics contribute code back to a variety
of libraries and software packages.

Current applications of Deep Learning in industry include voice recognition,
realtime translation, face recognition, video analysis, demographic
identification, and tagging. We expect this flourishing of new development to
continue over the next couple of years as new applications are discovered,
more data assets appropriate to Deep Learning become available, and GPUs become
even more affordable.

### Commercial Uses of Deep Learning

The current enthusiasm for Deep Learning was spawned by a 2012 paper from Google
^[http://arxiv.org/abs/1112.6209] describing how researchers trained a
neural network classifier on extremely large amounts of data to recognize cats
(many examples used for illustration purposes in this paper is a nod to these
researchers) Since then, big players like Google, Microsoft, Baidu, Facebook,
and Yahoo! have been using Deep Learning for a variety of applications.

All of these companies have a natural advantage in creating Deep Learning
systems -- massive, high-quality datasets. They also have expertise in managing
distributed computing infrastructure, giving them an advantage in applying Deep
Learning techniques to real-world problems.

Google has published a follow-up to its 2012 paper
^[http://arxiv.org/abs/1309.4168], and is now using Deep Learning for
realtime language translation, among other things. Its recent announcement of a
realtime voice translation service on mobile phones is impressive both for the
functionality ^[Who hasn't wished for a real-life Douglas Adams Babel
fish?] and for the application architecture -- it runs on a standard smartphone.

Facebook formed FAIR,^[https://research.facebook.com/ai] the Facebook Artificial
Intelligence Research Lab, in 2013, and hired Yann LeCun, one of the pioneers of
Deep Learning research and a professor at NYU, as its director. FAIR has developed
face recognition technology that has been deployed into an application that allows
Facebook users to organize personal photos and share them with friends,
^[See https://research.facebook.com/blog/814042348693053/fair-opening-up-about-artificial-intelligence-and-facial-recognition/.]
and contributed to numerous open source and academic projects. FAIR operates
from Facebook's New York Astor Place office, Menlo Park, CA, and Paris.

Baidu hired Andrew Ng, coauthor of the Google cat paper and Stanford professor,
to lead their Deep Learning research lab out of Cupertino, CA. Baidu's
Minwa system is a purpose-built Deep Learning machine for object recognition.

Yahoo! is using image classification to automatically enrich the metadata on
Flickr, its social photo site, by adding machine-generated tags to every
photo. ^[though not without controversy: see
http://mashable.com/2015/05/21/flickr-auto-tagging-errors/]

The relationships between researchers and companies are complex because these
techniques emerged from a small and tight-knit research community that has
recently exploded into relevance, with this obscure research area becoming one
of the hottest areas for hiring among the Internet giants. Ideas that may have
begun at one institution show up in applications developed by another, and
people often move between institutions.

#### Deep Learning as a Service

If you are considering using Deep Learning but don't plan to develop and train
your own models, this section provides a guide to companies that offer services,
generally through an API, that you can integrate into your products.

##### Clarifai

Clarifai ^[http://www.clarifai.com/] is a New York-based startup that uses Deep
Learning to recognize objects in still images and video data. Clarifai's models
won the 2013 ImageNet competition.

Clarifai's API allows users to submit an image or video, then returns a set of
probability-weighted tags describing the objects recognized in the image or
video and the system's confidence in each tag. The API can also use an image
input to find similar images. It runs quickly; it is capable
of identifying objects in video streams faster than the video plays.

Founder Matthew Zeiler says, "Clarifai is building products that empower people
to understand the massive amounts of information they are exposed to daily,
making it easy to automatically organize, analyze, and share."

##### Dextro 

New York-based Dextro ^[https://www.dextro.co/] offers a service that
analyzes video content to extract a high-level categorization as well as a
fine-grained visual timeline of the key concepts that appear onscreen. Dextro
powers discovery, search, curation, and explicit content moderation for
companies that work with large volumes of video.

Dextro's models are built specifically for the sight, sound, and motion of
video; its API accepts prerecorded videos or live streams, and outputs JSON
objects of identified concepts and scenes with a timeline of their visibility and
how prominent they were onscreen. Dextro can output in IAB Tier 2, Dextro's own
taxonomy, or any partner taxonomy.

Dextro offers a great service for any company that has an archive of video
content or live streams that they would like to make searchable, discoverable,
and useful.

David Luan, cofounder of Dextro, describes it as follows: "Dextro is focused
on immediate customer use cases in real-world video, and excels at
user-generated content. Our product roadmap is driven by our users; we
automatically train large batches of new concepts every week based on what our
partners ask for."

##### CloudSight 

CloudSight ^cloudsightapi.com/] is a Los Angeles-based company focusing
on image recognition and visual search. Their primary offering is an API that
accepts images and returns items that are identified in those images. They also
use this API to power two apps of their own: TapTapSee, which helps visually
impaired uses navigate using their mobile phones, and CamFind, which is a mobile
visual search tool where users can submit images from their cameras as queries.

[[FIG-cloudsight]]
.CloudSight recognizes a puppy
image::figures/cloudsight_beagle.png[scaledwidth="90%"]

##### MetaMind

MetaMind ^[https://www.metamind.io/] offers products that use recursive neural
networks and natural language processing for sentiment analysis, image object
recognition (especially food), and semantic similarity analysis. MetaMind is
located in Palo Alto, CA, and employs a number of former Stanford academics,
including its founder and CEO, Richard Socher. It raised $8 million of capital
in December 2014.

##### Dato

Dato ^[https://dato.com/] is a machine learning platform targeted at data
scientists and engineers.  It includes components that make it simple to
integrate Deep Learning as well as other machine learning approaches to
classification problems.

Dato doesn't expose its models, so you are not always certain what code,
exactly, is running. However, it offers fantastic speed compared to other
benchmarks. It's a good tool for data scientists and engineers who want a fast,
reliable model-in-a-box.

##### LTU Technologies 

LTU Technologies ^[https://www.ltutech.com/] is an image search company founded
in 1999 that offers a suite of products that search for similar images and can
detect differences in similar images. For example, searching a newspaper page
from two different dates may reveal two advertisements that are similar except
for the prices of the advertised product. LTU Technologies' software is also
geared toward brand and copyright tracking.

##### Nervana Systems

Nervana Systems ^[http://www.nervanasys.com/] offers a cloud hardware/software
solution for Deep Learning.  Nervana also maintains an open source Deep Learning
framework called Neon ^[https://github.com/nervanasystems/neon], which they
describe as the fastest available. Notably, Neon includes hyperparameter
optimization, which simplifies tuning of the model. Based in San Diego, Nervana
was founded in April 2014 and quickly raised a $3.3 million Series A round of
capital.

##### Skymind

Skymind ^[http://www.skymind.io/] is a startup founded by Adam Gibson, who wrote the open source package
DeepLearning4j ^[http://deeplearning4j.org/]. Skymind
provides support for enterprise companies that use DeepLearning4j in commercial
applications. Skymind refers to itself as "The Red Hat of Open-Source AI for
Enterprise."

Unlike other startups listed here, Skymind does not provide service in the form of an
API. Rather, it provides an entire general-purpose Deep Learning framework to
be run with Hadoop or with Amazon Web Services Spark GPU systems. The framework
itself is free; Skymind sells _support_ to help deploy and maintain the
framework. Skymind claims that DeepLearning4j is usable for voice-to-text tasks,
object and face recognition, fraud detection, and text analysis.

##### Recently acquired startups

There have recently been many acquisitions of startups using Deep Learning
technology. These include Skybox Imaging (acquired by Google), Jetpac (also
acquired by Google), Lookflow (acquired by Yahoo!), AlchemyAPI (acquired by
IBM), Madbits (acquired by Twitter), and Whetlab (also acquired by Twitter).

The challenges for independent Deep Learning startups include attracting the
necessary talent (generally PhDs with expertise in neural networks and computer
vision), accessing a suitably large and clean dataset, and finally figuring out
a business model that leads to a reasonable monetization of their technology.
Given these challenges, it's not surprising that many small companies choose to
continue their missions inside of larger organizations.

##### Startups Applying Deep Learning to a Specific Domain

Many entrepreneurs are excited about the potential of Deep Learning and are building
products that have only recently become possible because of the accessibility of these
techniques.

##### Healthcare

Healthcare applications push the boundaries of machine learning with large
datasets and a tempting market with a real impact.

Enlitic ^[http://www.enlitic.com/], a San Francisco-based startup, uses Deep
Learning for medical diagnostics, focusing on radiology data. Deep Learning is
fitting for this application because a Deep Learning system can analyze more
information than a doctor has ready access to, and may notice subtleties that
are not clear to humans.

Atomwise ^[http://www.atomwise.com] is a Canadian startup with offices in San
Francisco that focuses on using Deep Learning to identify new drug candidates,
and Deep Genomics ^[http://www.deepgenomics.com/] focuses on computational
genomics and biomolecule generation.

For quite a long time, analysis of text records has been used to improve the
patient experience and expand the knowledge available to doctors and nurses.
However, that information is severely limited in scope and detail, simply as a
result of being a burden to maintain for doctors. The ability to
automatically encode the information in images such as x-ray and MRI scans
into these records would provide a valuable additional source of data,
without placing an added burden on the healthcare practitioners.

##### Security

The security domain offers subtle pattern recognition problems. Is this server
failing? Is that one being compromised by unusual external activity? Deep
learning is a fantastic mechanism for this sort of problem since it can be
trained to understand what "normal operating conditions" are and alert when
something deviates from it (regardless of whether the operators knew whether to
intentionally put in a rule or heuristic for it).  For this reason, neural
networks have even been used as a backup system to monitor the safety of nuclear
facilities.  As a result, there has been progress in both academic research
^[http://dl.acm.org/citation.cfm?id=2713592] and industry.

Canary ^[http://canary.is/] offers a home security device that alerts the
customer's mobile phone when unusual activity is detected in the home.

Lookout ^[https://www.lookout.com] offers an enterprise mobile predictive security
solution that identifies potential problems before they manifest.

##### Marketing

Marketing is an obvious application domain for image analysis. Current media
analytics products are generally limited to analyzing text data, and being able
to extend that analysis to images has real value.

One product in this space comes from Ditto Labs ^[http://ditto.us.com], a startup based in
Cambridge, MA. Their analytics product scans Twitter
and Instagram photos for identifiable products and logos, and creates a feed for
marketers showing brand analytics on social media. Marketers can track their own
brands or competitors' in a dashboard, or use an API.

Being able to identify the demographics of consumers is another interesting
application that is already in the wild. Kairos ^[http://www.kairos.com]
specializes in face recognition and analysis in images and video. Their SDK and
APIs allow customers to identify individuals by face as well as estimate the
demographics of unknown faces (gender, age, emotions, engagement). Finally, they
offer a crowd analysis product that automates crowd analytics.

##### Data Enrichment

Document analysis firm Captricity ^[https://captricity.com/] offers a product
that helps established companies with information in paper format, such as
insurance companies, convert this information into useful and accurate digital
data. Captricity uses Deep Learning to identify the type of form represented in
a scan -- for example, a death certificate -- and to recognize when form fields are
empty.

### Neural Network Patents

Our review of patents in the area does not reveal any dominant patent holders or
limiting patents in the field. Rather, the field seems scattered with only a
handful of patents issued to a small number of patentees. Given that the
quantity and quality of data fed into the neural network is a major factor in
the utility of the system, it is not surprising that there are few key patents
in the area of Deep Learning.

There is one notable exception here: NEC Laboratories has received a number of
Deep Learning patents, with many focused on visual analytical applications and
some applications in text processing. As a few examples, NEC holds US Patent No.
8,345,984, covering methods for recognizing human actions in video images using
convolutional neural nets; US Patent No. 8,582,807, dealing with gender
classification and age estimation of humans in video images; US Patent No.
8,234,228, covering methods for unsupervised training of neural nets; and US
Patent No. 8,892,488, focusing on document classification. 

### Open Source Neural Network Tools

The open source landscape for neural network tools is quite vast. This stems
primarily from the fact that this is a very active academic field and is
constantly changing. As a result, the only tools that can stay on top of the
trends are those that are open source, have a thriving community, and cater to
both users and the academics leading the advances in the field. We provide a
survey here of both the tried and true libraries and the ones that the community
is the most excited about at the time of writing; bear in mind, though, that the
landscape is constantly changing.  New libraries are sure to be introduced to
the community, although they will mostly be specially built for a particular
purpose or built on one of these technologies.

##### Theano

Theano is a Python-based open source tensor math library that couples tightly with
`numpy` and provides mechanisms for both CPU- and GPU-based computing. As a
result, `theano` is used quite heavily when building new types of neural networks
or building existing networks from scratch: code can be prototyped quickly
yet still run at GPU speeds.

While this library is extremely powerful, it is often overly general for those
just getting into the field of neural networks. However, since most other
Python neural network libraries use `theano` in the background, it still can be
an incredibly important tool to understand.

##### PyBrain2

PyBrain2 is another open-source Python neural network library focused on
simplicity. It is provided as a layer on top of `theano`, with a simple
pipeline for creating neural networks from already well understood pieces. As a
result, a convolutional neural network can be quickly and easily deployed using
this system.

However, `pybrain2` does not completely hide the internals of the models you are
creating, and as a result it is used quite frequently in academic work when
experimenting with new network designs or training procedures. As a result, its
API is rapidly evolving, with new features being added regularly.

`pybrain2` is a good choice for beginners to intermediate users who want
to potentially work with the internals of their network but don't want to
reimplement existing work.

##### Keras

Keras, the final open-source Python library we will look at, is the simplest of
all the options, providing an extremely simple and clean API to quickly and
easily create networks with well understood layers. While it does provide
mechanisms to see the internals of the network, the focus is on doing a small
set of things correctly and efficiently.

As a result, many of the recent innovations in neural networks can be easily
implemented using `keras`. For example, an image captioning network can be
implemented and trained in just 25 lines of code!
^[http://keras.io/examples/] This makes `keras` the best choice for
Python developers looking to play with neural networks without spending too much
time working on the internals of the model.

##### caffe

Caffe is a C++ library created at UC Berkeley, released under the 2-clause BSD
license, with a Python library included. It is quite fully featured and is
generally used as a standalone program. In it, most of the common neural network
methods are already implemented and any customized model can be created by
simply creating a YAML configuration file. In addition, as mentioned previously,
many pretrained models are provided through Caffe's "model zoo,"
^[https://github.com/BVLC/caffe/wiki/Model-Zoo] which makes this a good
option for those wishing to use preexisting models. Finally, `caffe` has been
accepted by NVIDIA as an official neural network library and as a result is
_very_ optimized to run on their GPUs. ^[Make sure to download the
version of `caffe` from NVIDIA's GitHub repo, as well as their specialized math
libraries, in order to take advantage of these optimizations.]

While `caffe` is quite fast to use and to run, especially if using a pretrained
network, the API is quite unfriendly and installing it is known to be difficult.
The main source of pain with the API is understanding how custom data should be
represented for use with the standalone application. Furthermore, `caffe`'s
documentation is quite lacking, which makes understanding how to create the YAML
configuration for a new network tedious. This lack of documentation carries
over to the Python library, which is just a thin wrapper over the C++ library.

In the end, even though `caffe` is quite a fully featured and robust toolkit, it
still feels very much like academic code.

However, once the dataset is properly formatted and the YAML configuration is
error free, `caffe` is quite fast and provides all of the benchmarking one would
expect from such a fully featured application.

##### Torch

Torch is another neural network library released under the BSD license, written
in Lua. At its core, Torch is simply a powerful tensor library (similar to
Theano); however, a system of modules has been made around it, creating a
friendly and simple ecosystem for applications in neural networks.
^[https://github.com/torch/torch7/wiki/Cheatsheet] Torch has been
supported by many of the academic groups involved in neural network research, as
well as many of the industrial research groups, such as Google and Facebook.

Since most usage of Torch is through various community-created modules,
documentation can be hit or miss. However, the community has a strong emphasis
on good documentation and sticking with Torch's strict and clean design
practices. In fact, many new neural network libraries (written in Lua or in
other languages) have been adopting the Torch paradigm for creating neural
network pipelines. In this pipeline, a model definition is created in Lua, as
well as a separate dataset definition. This is then run through a training
script, which combines both of these definitions in order to create a trained
model on the given dataset. This modularization makes the resulting code very
robust and helps with maintaining models as datasets change.

The biggest downside to Torch is that it is in Lua. While Lua is a fantastic
language that is gaining traction within the academic world (particularly
because of the extremely performant LuaJIT), it may not be easy to incorporate
into existing deployment strategies or infrastructures. However, the community
is actively trying to combat this by providing AWS images in which Torch
is ready to use and helpful documentation giving insight not only into how
Torch works but why Lua was chosen and how to use it effectively.

##### Brain.js and Convnet.js

Brain.js is a JavaScript neural network library released under the MIT license.
The library offers a simple and expressive way of defining, training, and
evaluating neural networks. Available both as a client-side and a server-side
tool, it provides a lot of useful possibilities when integrating neural networks
into a larger web service. It does not carry many of the out-of-the box features
that many other neural network libraries will give you, but instead has the
bare-bones algorithms implemented to get you going. Being written in JavaScript,
Brain.js is a great way to evaluate neural networks, but it lacks the optimizations
(such as GPU integration) to do efficient training for larger applications.

Convnet.js is another JavaScript neural network library, built by Andrej Karpathy and released under the MIT
license. It brought Deep Learning into the browser and provides more of the
technical specifications an AI expert would expect in a library. Written to
support convolutional neural networks, the library has many of the common
modules (e.g., fully connected layers and nonlinearities) and cost functions
built in. Convnet.js has particularly been a boon for visualizations and demoing,
helping people learn about and understand neural networks simply and in their
browsers. While the library can be used to train neural nets, again it is not
optimized for production systems; however, it does serve its purpose as a tool ready for
browser deployment.

What makes these libraries exciting is that pretrained models can be put into
the browser and evaluated live on the client's machine. Also, as robust
JavaScript implementations of neural networks, they have a value due to the
ubiquity of JavaScript applications on the Web.
