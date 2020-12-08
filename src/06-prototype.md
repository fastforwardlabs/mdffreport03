## Prototypes: Pictograph and Fathom

While researching and tinkering with neural networks, there was a surplus of
cool applications and interesting problems appropriate for exploring that
practical side of this technology. The major drawback was that many compelling
ideas required large, unavailable (or proprietary) datasets in order to get the
accuracy and performance we desired. In the end, we used images gathered from
users on Instagram and a model available through caffe to perform object
recognition on user-uploaded photos.

### Pictograph and Fathom

For this project, we built two prototypes powered by the same backend. Our public prototype, Pictograph, uses image object recognition to classify a user's Instagram photos. Our subscriber-only prototype, Fathom, allows you to explore our Instagram data set through computer identified labels and categories. Together the prototypes demonstrate what is possible with out of the box ImageNet image classification systems.

This type of technology is becoming more prevalent in
larger organizations with large datasets. Even though high quality pre-trained
models are becoming more and more available, the utility of these has not quite
yet been shown to end users. Pictograph and Fathom show the new photo exploration possibilities of these models.

#### Backend

The core of the prototypes is the pre-trained `googlenet`
^[http://arxiv.org/abs/1409.4842] model from caffe's model zoo. Out of
the box, this model has the ability to classify images over 1000 possible
labels.  These labels are all nouns taken from the WordNet
^[https://wordnet.princeton.edu/] corpus and include things such as:
lampshade, flatworm, grocery store, toaster and pool table.  Furthermore, this
model is provided under an unrestricted license
^[https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet] and
can be easily used with caffe (see <<caffe>>).

[[FIG-sample-tree]]
.Sample labels from the model
image::figures/05/labels-01.png[scaledwidth="100%"]

Using Caffe's python bindings, pycaffe, we were able to create a fully
featured web application using tornado as our web server.  Once the application
is started, we load up the googlenet model into GPU memory, as well as any other
data that is necessary to do label disambiguation.  By pre-loading the model and
all auxiliary data needed, we can easily take HTTP requests via tornado
requesting an image classification and route it to the GPU with little overhead.
The resulting operation takes about 300ms per image.  Having the GPU ready for
this calculation is critical as it can take up to 7 seconds per image if the
system is operating in CPU-only mode.

For further optimization, we cache most of the image classification results when
the user first authenticates in.  On authentication, we fetch all of the user's
images, classify them, and insert them into a rethinkDB
^[http://rethinkdb.com/] instance.  RethinkDB was chosen as our backing
database due to its ease of use, robustness, and very clean API.


#### Dealing with Low Confidence

In order to deal with potentially low confidence results given by the neural
model, we chose to use the confidence levels over the labels in order to
hierarchically cluster the labels. We can do this by taking the actual labels
(i.e., whale, farm, Egyptian cat) and use them as leaves when building a tree from
their hypernyms ^[A hypernym is "a word with a broad meaning that more
specific words fall under; a superordinate. For example, color is a hypernym of
red."]. This means that we have a tree with labels "beagle", "golden retriever",
and all other dogs, connected under the label dog. And dogs and cats are
together under the label "domestic animal", all the way up until we reach the
most general label "entity".

With the tree built up, we gain the ability to do many sorts of operations over
sets of label predictions. For example, with a single label prediction we can
propagate the confidence levels up by setting each hypernym's value to the
average confidence of its children's values. With this, if we have several
medium-confidence predictions over various types of dogs, the hypernym "dog"
will have a high weight value. This basic ability allows us to also do more
complicated operations. For example, if we wanted to compare two users we can
take the sum of all of their respective image predictions and then take the dot
product of those vectors. Putting this resulting vector through the hypernym
tree will give hypernyms that both users take pictures of. Similar operations
can be created to see dissimilar images and to do them with emphasis on the
extremely similar/dissimilar labels (Anne takes pictures of dogs and Bill does
not) or just asymmetries (Bill takes more pictures of dogs than Anne).

![Sample tree showing confidence propagation.](figures/05/tree-01.png)

This step of augmenting the label predictions through a clustering scheme is
incredibly important when dealing with potentially low confidence results.
Since the data being sent to the model is often a lot more noisy than the
original training dataset (which contained mainly stock photography) the actual
accuracies we achieve on live data will be substantially less than what was
advertised for the training data. Instead of hand-annotating live data in order
to fine-tune the network with more realistic data we realized that, while
sometimes the confidence in a classification was low, the network was generally
pointing towards the correct concept. That is to say, even though there was no
high confidence label for a given image of a dog, all of the dog-like labels had
higher confidences than non-dog labels.

In doing the clustering, we are able to hide the low confidence in some
classifications while still giving the user useful results. In the end, the user may
not care whether we classify what _type_ of dog is in the image but be more
interested simply that there is a dog. Furthermore, having the taxonomy over
the possible labels introduces a novel way of navigating through the dataset and
gaining a deeper understanding of the landscape of images being taken.

### Design and Deep Learning

By enabling increasingly accurate image object recognition and natural language
processing, deep learning has the potential to open up new design possibilities
for social web apps. The default organizational view for most social networks
(e.g., Facebook, Twitter, Instagram) is the reverse-chronological feed, where
posts are organized into topic-agnostic content blocks. Deep learning, by
allowing designers to take the topic and sentiment of user posts into account,
could enable new forms of organization and allow for designs that adapt
themselves to better support different subjects and moods.

#### Beyond the Feed

In our prototypes we demonstrate how image object recognition can open up new
paths of exploration in Instagram. While Instagram does have the capacity to
group images together by category based on hashtags, uneven and idiosyncratic
use of tags limits the effectiveness of this system-wide.  ^[While there
aren't numbers for Instagram, on Twitter hashtags are only used on 24% of tweets
https://blog.bufferapp.com/10-new-twitter-stats-twitter-statistics-to-help-you-reach-your-followers.]
To have their pictures of a dog appear in the dog hashtag category, an Instagram
user has to add '#dog' to each one of the relevant photos. In Pictograph and Fathom this
classification is taken care of, and thanks to our use of hierarchical tree
classification that category is also automatically included under parent
categories such as 'animal'. Because we use image object recognition we don't
have to rely on users to input an image classification and can therefore include
a much larger number of public images from the network in the relevant category page.

![The dog category page on Fathom](figures/06/fathom.jpg)

While our prototype focuses on the classifying possibilities of image object
recognition. Other techniques powered by deep learning, such as natural language
processing, could perform similar classifying and tagging operations for
text-heavy posts. The capability to infer this kind of metadata from currently
untagged user generated posts could greatly increase our ability to discover
content by topic -- a method of exploring that is intuitive to humans but up to
now has been difficult to support at scale. 

The information unlocked by deep learning will have to be integrated thoughtfully into current app design conventions. Better category classification could allow us to move away from reverse-chronological feeds -- lengthening the lifespan of useful posts. The new info could also be used to improve the relevance of search results on social networks. These same qualities could also alter the moods of networks, however. Users may have become accustomed to old posts being buried in the stream or not readily visible to people outside their friends. For example, while most
Instagram photos are public, for the non-celebrity the network can still feel like a very intimate
place. If user photos are made more visible by improved classification
features, such as the category pages in Fathom, that feeling of intimacy may be lost. Product creators will need to balance the increased access that deep learning can enable with user expectations.

While the ability to explore through category pages in Fathom can be seen
roughly as an improvement to tagged category pages, the 'Pictograph' display,
where a user's top four most photographed categories are arranged as a
proportional treemap, suggests some of the new visualizations that will be
available as these technologies evolve.

![A Pictograph user's pictograph](figures/06/pictograph.png)

The pictograph visual moves toward saying something about the user's personality
based on classification. This possibility is only opened up when we have access
to information about the content of a user's images. ^[Again, hashtags could
conceivably be used to perform a similar function, but real-world use of them is
rarely consistent enough to support such a project.]

![Pictograph uses image object recognition to perform the kind of evaluation of a user's profile we usually associate with a human (or anthropomorphic robot), on-demand and at a large scale.](figures/06/robot-01.png)

Paired with a hierarchical classifying system, this sort of classification could
allow designs to adjust to support a user's interests, or to build specially
designed experiences around specific categories. Graphic designers who have
access to the mood and content of a piece can design to support and enhance
those elements. In the past this took the form of a designer being given a
specific piece, in the future it might involve designing for specific mood and
topic targets -- where the content will then be matched to the design by a set
of algorithmic classifiers. In this way, deep learning could support a return to content-specific design at a scale not previously possible.

### Failed Prototypes

One common theme from people working with neural networks is the amount of
failure when trying to build new systems. On the way to creating the Pictograph
and Fathom prototype, we tried many things that simply did not work for one
reason or another. One reason for this was the time limitation -- as described
in the <<devtimeline,development timeline>> section, creating new models can be
time consuming. Another reason for this is availability of data -- since we
don't have our own in-house datasets outside of data partnerships, finding the
interesting and compelling problems for the datasets we have can be a challenge.

Below is a short list describing of some of the prototype ideas we were excited
about but did not work.  The recurring theme is the availability of data --
many of the ideas were completely technically feasible however require access to
clean data that was not easily attainable.  Constant discussions were had
regarding how to clean the data that we _could_ get, from hand labeling to using
Amazon Turk, however the cost of these methods couple with still not knowing if
they would work turned us away from them. This serves as a testament to how
important data quality is when creating neural systems.

#### Giphy

Giphy ^[http://giphy.com/] is a search engine for animated GIFs.
Animations are human annotated with tags that refer to the content of the
animation with tags varying from "cat" to "cute" or "shock".  Our goal was to
predict tags given an animation using video classification methods
^[http://cs.stanford.edu/people/karpathy/deepvideo/].  However, we faced
many challenges when trying to create such a system.

The first challenge was dealing with the variability in GIF data.  GIF's have
multiple standards that all deal with keyframing differently.  This makes it so
that when you are parsing an animation, great care must be taken to make sure
the representations you have for each frame are correct and with minimal noise.
In addition to simply extracting frames, many decisions had to be made for
creating the neural system where framerates could be so variable.  In general
neural-video processing, it videos can be assumed to run at a standard
framerate.  GIF's, on the otherhand, operate at a wide variety of frame rates
(from 0.5 frame/second to 60 frames/second).

More importantly, however, was dealing with the quality of the tags.  For
typical image classification problems, the labels that are being extracted are
very simple.  Even when asking a human to do the comparable task, it is very
simple to answer whether there is a dog in an image, but it is much more a
matter of opinion whether an image portrays "shock" or "happy".  For a neural
method to be able to make this identifications we need both a large set of
examples in addition to a deep network in order to extract the relevant
features.

In the end, we were not able to train a large enough model with the given
dataset given our timeline.  As discussed in the <<devtimeline,development
timeline>>, neural networks have a large upfront time cost where you may
understand what may be necessary for the model to work but it takes quite a long
time to validate that hypothesis.


#### Nutrition

Seeing the wealth of images online of food, we also had the idea of creating a
system which would take in an image of a dish and output the nutritional content
of a dish.  The plan was to find a repository with the nutritional content of
various dishes, search for the dish on flickr ^[http://flickr.com] and
train out model using this data.

The main problem with this attempt was the quality of the data.  Here, since we
were joining two different datasets (the nutritional content and the images of
the food) the possibility for bad data was multiplied.  Firstly, nutritional
data for dishes is not very available.  There are many sites that claim to
provide the nutritional content of dishes (along with their recipes), however
there are _large_ discrepancies between what the sites report.  This is probably
because there are multiple ways to make a given dish, all of which contain
slightly different ingredients and different quantities of them.  In the end,
one set of data was chosen as the "ground truth" simply because of the breadth
of dishes that had data (including calorie, fat, protein, carbohydrates).  This
would also fare as a good benchmark for whether the system could work at all and
whether it was worth it to put more time and resources in a potentially more
accurate nutritional database.

Once we had some semblance of nutritional data we had to acquire images of each
dish.  The idea was to search flickr for the dish names along with certain
phrases to select for images of food (such as "dish" and "food").  We keep the
dish in our dataset if we were able to find 1,000 relevant images.  However,
upon doing a data quality check, it was evident that many pictures of a dish are
of people _cooking_ the dish as oppose to the final dish itself.  This meant
that many images of "burritos", for example, were actually of a dish of rice or
avocados that were being prepared for the final dish.  In addition to simply
being the wrong image for a given dish label, these images had overlap with
other dishes (for example a salad) which further confused the neural network's
internal representation of the data.

In the end, data quality was the primary reason why this experiment failed.
With images that were labeled under burrito that looked like a salad, pictures
of a cutting board with a knife or a steaming pot for many of the dishes, and
other such examples, there was no coherent features that could be extracted to
discriminate between one dish or another.  This problem persisted whether we
tried to train a new neural network or frame it as a transfer task; as we
trained the system we would constantly get wildly fluctuating results, an
indication that the model couldn't converge on relevant image features.
Furthermore, we found out later that it is indeed possible -- several days after
we put this experiment away Google announced it had done a similar task of
assigning caloric value to the image of a dish
^[http://www.popsci.com/google-using-ai-count-calories-food-photos].
