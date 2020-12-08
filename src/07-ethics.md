## Ethics of Deep Learning

When we create a data product, we must consider not just the mathematical and
computational aspects of the algorithm, but what the product experience of that
algorithm feels like and the opportunities for it to go awry. Considering the
worst-case scenario is not just something for Philip K. Dick and Ursula LeGuin; it
must also be done conscientiously by engineers and product designers.

Deep Learning requires some creative thinking, for a number of reasons.

### Uninterpretability

With Deep Learning, feature engineering is left to the algorithm. Therefore, you
are never entirely certain what a specific feature represents, or even
ultimately why the algorithm assigned the label that it produced. In machine
learning we call this kind of algorithm _uninterpretable_.

Back in Section 2, we saw that neural networks learn by adjusting the weights and
biases that connect and activate neurons. This simple strategy gives us great
computational power, but reveals to us nothing about what any of these numbers
or connections actually _mean_. That is, the underlying heuristics are nearly
impossible to interpret. If the network encoded a feature based on race, there
would be no easy way to figure this out or correct it. It is analogous to how
neuroscientists and philosophers can't discover properties of consciousness by
simply looking at brain scans. This is a particular concern for companies
working in highly regulated industries.

These risks are on the radar of many government and business leaders. Consider
White House Counselor John Podesta warning students at the UC Berkeley School of
Information, "We have a strong legal framework in this country forbidding
discrimination based on [race, ethnicity, religion, gender, age, and sexual
orientation] criteria in a variety of contexts. But it's easy to imagine how big
data technology, if used to cross legal lines we have been careful to set, could
end up reinforcing existing inequities in housing, credit, employment, health,
and education."
^[http://m.whitehouse.gov/sites/default/files/docs/040114_remarks_john_podesta_big_data_1.pdf]

This is not a purely theoretical concern. Google's image recognition software
recently tagged two people of color as "gorillas," and labeled a photo of the
concentration camp at Auschwitz as a "jungle gym".
^[http://www.theguardian.com/technology/2015/jul/01/google-sorry-racist-auto-tag-photo-app]
Unfortunately, due the complexity of the system, the quick fix for the former
offensive result was to simply eliminate the "gorilla" tag from the system entirely.

Uninterpretable systems are also vulnerable to propagating biases in the
original source data. In one example, an ad server was observed advertising more
higher-paying jobs to men than to women.
^[https://www.andrew.cmu.edu/user/danupam/dtd-pets15.pdf] It's extremely
doubtful that any engineer designed that interaction, but it's important to
acknowledge the possibility of a system that uses gender, ethnicity, or
socioeconomic status as a proxy for potential success in a highly abstracted
feature space.

There is a further threat of having a system that does not have an inherent
bias, but merely misinterprets a profile due to constraints in understanding
context (see Section 2 for more on this). Take as an example those who are
unjustly investigated and sometimes even prosecuted for crimes due to perceived
"involvement" because they have family or friends that are involved in criminal
activity. We see this as unethical because one cannot help where one is born,
and thus one's family and, to some extent, choice of friends. However, the issue
here is that the data and profile surrounding the innocent person look a lot
like those of the individuals involved in the crimes. Again we find a scenario
in which neural networks could fundamentally misinterpret someone whose profile
_looks like_ it matches a certain classification. The consequences of such a
misinterpretation must be considered carefully as delivery of such a system
proceeds.

Some data leaders have recently proposed a method of "bias testing" that would
develop metrics for evaluating and monitoring the likely bias of a deployed
model. This is an active area of discussion and research.

### Edge Cases: Liability and Error

Even with a model that is mathematically near perfect, there may be product
consequences. Current Deep Learning systems typically achieve accuracy rates
from 80-95%. But what about the >5% of results that are wrong?

Some examples of this kind of unfortunate edge case are Google's data-driven
overestimation of the scope of the influenza risk during the winter of
2012-2013, which left hospitals and clinics underprepared,
^[http://www.nature.com/news/when-google-got-flu-wrong-1.12413]
Facebook's Year in Review showing a user photos of his recently deceased
daughter,
^[http://www.theguardian.com/technology/2014/dec/29/facebook-apologises-over-cruel-year-in-review-clips]
and neural networks that mislabel images due to human-imperceptible distortions.
^[http://arxiv.org/pdf/1312.6199v4.pdf]

As we enter a world where more neural networks are doing work across
our business, state, and private ecosystems, we must take seriously the
potential of an edge case causing serious harm. We are already seeing businesses
forced to apologize publicly or settle cases due to social or emotional
traumas inflicted upon individuals. We expect these incidents to occur more frequently
as more products enter the market. Most of these technologies have obvious
benefits -- medical diagnoses, predictive traffic modeling, maintenance
prediction -- but the cost of an incorrect outcome may be monumental and it is
unclear where the liability lies.

#### Unethical Applications

As this report is being written, The Future of Life Institute has already
received signatures from over 1,000 robotics and AI researchers petitioning for
a global ban on the development of weaponized AI. This intersects directly with
the development of Deep Learning, since computer vision and classification would
both be major components of any intelligent weaponry. While these technologies
could be very good at their intended purposes and sometimes put people out of
harm's way, the potential to do bad seems high. This is an ongoing subject
of debate among academics and people in the field.

If we zoom out from this stigmatized issue, we can find other application areas
that carry similar face-value problems. Take for instance the possibility of a
neural network that could profile online users or customers as vulnerable sales
targets for high-interest loans. Given the data financial institutes have, this
would be a fairly straightforward application for training a neural network.
Yet, we recognize this as an unethical business practice that takes advantage of asymmetries in
information and unfairly leverages socioeconomic data to place emotional pressure on
people who want a better life. There have already been applications showing
the possibility of this type of threat, such as Google advertising jail bonds to people
with traditionally African-American names
^[http://techcrunch.com/2013/02/05/googles-unintentionally-racist-ads-probably-have-awful-psychological-impacts/]
and Facebook's recent patent to help lenders do loan discrimination based on the applicant's social
network connections.
^[http://venturebeat.com/2015/08/04/facebook-patents-technology-to-help-lenders-discriminate-against-borrowers-based-on-social-connections/]

We can find many examples where data is available to train a neural network to
do something of questionable moral value. Price discrimination could easily be 
implemented by learning from online profiles and guessing the maximum price to
display to a consumer. More insidious practices can be imagined where neural
networks are trained to identify people taking part in protests, to
deanonymize account information, or to discover personal vulnerabilities. In
each of these cases, the application of a neural network provides one party
with exploitative influence over another due to asymmetric access to or
ownership of data.

#### What You Can Do

There are a few strategies used in the industry for avoiding these issues. The
most basic and important of these strategies is taking appropriate time to
consider the impact your system may have on a user -- that is, having a good
answer to the question, "What does it mean to a user if our system is wrong?"
Preemptively recognizing edge cases and having an increased awareness of failure
will help prevent surprises and improve the expectations of the users. It is
expensive to train and retrain Deep Learning models, so reasonable forethought
can save time and energy later.

Of course, these limitations cannot always be known in advance, but prepared
engineering teams will always test their systems against diverse inputs. This is
where considering a combination of edge cases and consequences alongside a
robust testing protocol goes a long way. Testing your data against the
unexpected user, or even the contrived worst-case scenario, will reveal a lot
about the expected performance of the system and the potential errors. This is
all good hygiene for robust systems.
