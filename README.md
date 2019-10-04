# Traffic-Signs-Classification

<h2>Problem Definition</h2>
<p>This Group Project was implemented as part of the course: Big Data Management offered by the M.Sc in Data Science of the Department of Informatics, School of Sciences, National and Kapodistrian University of Athens.</p>
<p>The problem that this repository is called to solve is the classification of traffic signs. The traffic signs are depicted as images and the approach that is used is the Convolutional Neural Networks.</p>

<h2>The Dataset</h2>
<p>The dataset that is used is the pickled German Traffic Signs dataset and can be found <a href="https://www.kaggle.com/imadmoussa/traffic-signs">here</a>. It contains 43 different categories of traffic signs which correspond to the different classes of the classification problem. The dataset contains 3 different files e.g. train.p, valid.p and test.p that are used to train and validate the model and finally test whether the final classification model is capable of generalization.</p>

<p>The training set contains 34.799 images, the validation set 4.410 and the test set 12.630. Each image is in the form of a 32x32x3 matrix where 32x32 corresponds to the width and height of the image and the 3rd dimension to the 3 color channels. After some exploration and visualization, it can be observed that the classes do not contain the same number of training examples and thus this would introduce bias in models resulting to bad predictions.</p>

<h2>Preprocessing and Augmentation</h2>
<p>There are many cases in which a dataset is not capable of providing enough good training patterns to produce a good classification model. Different types of datasets require different handling to address problems that may arise, even before beginning the training phase. For example, when an image dataset contains only nearly perfect photographs of the object of interest it may produce a model that is not capable to recognize the same object in a low quality or a rotated photograph. To avoid this problem a preprocessing step is carried out prior to the training phase, in order to provide a variety of images that introduce as many views of the object of interest as possible.</p>

<p>Another important problem that is tackled with preprocessing and augmentation is class imbalance. As mentioned, neural networks are susceptible to producing biased models when the dataset contains classes with different numbers of training patterns for each class. Quantitively, a neural network when trained with an unbalanced dataset, “learns” about the classes with the most training patterns and thus it is possible that the probability of misclassification for images of classes containing few patterns is high. To avoid this, more data of these classes is needed. But acquiring new data is not always cheap. Luckily, given an image, one can apply countless different filters to produce new images to add to the dataset. This procedure is called data augmentation and it is a very efficient way to deal with the class imbalance problem. Examples of filters that can be used are, adding noise, smoothing the edges, rotating the image and many others.</p>

<p>As mentioned, the dataset that is used for the training phase in this project is highly unbalanced and, in many cases, different views of certain images could provide new patterns to improve the accuracy of the model on the testing set. The dataset originally contains 34.799 images of 43 classes. The augmentation process that was followed, which will be explained analytically below, enhanced the dataset to 258.227 images which is about 7.5 times the starting size.</p>

<p>The different filters that were used to achieve that are:
<ul>
  <li>Bilateral Filtering</li>
  <li>Gaussian Blur</li>
  <li>Rotation (±10 degrees)</li>
  <li>Translation</li>
  <li>Perspective Transformation</li>
</ul></p>
