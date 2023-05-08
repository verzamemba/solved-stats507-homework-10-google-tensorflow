Download Link: https://assignmentchef.com/product/solved-stats507-homework-10-google-tensorflow
<br>
<h1>1         Warmup: Constructing a 3-tensor</h1>

You may have noticed that the TensorFlow logo, seen in Figure 1 below, is a 2-dimensional depiction of a 3-dimensional orange structure, which casts shadows shaped like a “T” and an “F”, depending on the direction of the light. The structure is five “cells” tall, four wide and three deep.

Create a TensorFlow constant tensor tflogo with shape 5-by-4-by-3. This tensor will represent the 5-by-4-by-3 volume that contains the orange structure depicted in the logo (said another way, the orange structure is inscribed in this 5-by-4-by-3 volume). Each cell of your tensor should correspond to one cell in this volume. Each entry of your tensor should be 1 if and only if the corresponding cell is part of the orange structure, and should be 0 otherwise. Looking at the logo, we see that the orange structure can be broken into 11 cubic cells, so your tensor tflogo should have precisely 11 non-zero entries. For the sake of consistency, the (0<em>,</em>3<em>,</em>2)-entry of your tensor (using 0-indexing) should correspond to the top rear corner of the structure where the cross of the “T” meets the top of the “F”. <strong>Note: </strong>if you look carefully, the shadows in the logo do not correctly reflect the orange structure—the shadow of the “T” is incorrectly drawn. Do not let this fool you!

<strong>Hint: </strong>you may find it easier to create a Numpy array representing the structure first, then turn that Numpy array into a TensorFlow constant. <strong>Second hint: </strong>as a sanity check, try printing your tensor. You should see a series of 4-by-3 matrices, as though you

Figure 1: The TensorFlow logo.

were looking at one horizontal slice of the tensor at a time, working your way from top to bottom.

<h1>2         Building and training simple models</h1>

In this problem, you’ll use TensorFlow to build the loss functions for a pair of commonlyused statistical models. In all cases, your answer should include placeholder variables x and ytrue, which will serve as the predictor (independent variable) and response (dependent variable), respectively. Please use W to denote a parameter that multiplies the predictor, and b to denote a bias parameter (i.e., a parameter that is added).

<ol>

 <li><strong>Logistic regression with a negative log-likelihood loss. </strong>In this model, which we discussed briefly in class, the binary variable <em>Y </em>is distributed as a Bernoulli random variable with success parameter <em>σ</em>(<em>W<sup>T</sup>X</em>+<em>b</em>)<em>, </em>where <em>σ</em>(<em>z</em>) = (1+exp(−<em>z</em>))<sup>−1 </sup>is the logistic function, and <em>X </em>∈ R<sup>6 </sup>is the predictor random variable, and <em>W </em>∈ R<sup>6</sup><em>,b </em>∈R are the model parameters. Derive the log-likelihood of <em>Y </em>, and write the TensorFlow code that represents the negative log-likelihood loss function. <strong>Hint: </strong>the loss should be a sum over all observations of a negative log-likelihood term.</li>

 <li><strong>Estimating parameters in logistic regression. </strong> contains four Numpy .npy files that contain train and test data generated from a logistic model:

  <ul>

   <li>logistic npy : contains a 500-by-6 matrix whose rows are the independent variables (predictors) from the test set.</li>

   <li>logistic npy : contains a 2000-by-6 matrix whose rows are the independent variables (predictors) from the train set.</li>

   <li>logistic npy : contains a binary 500-dimensional vector of dependent variables (responses) from the test set.</li>

   <li>logistic npy : contains a binary 2000-dimensional vector of dependent variables (responses) from the train set.</li>

  </ul></li>

</ol>

The <em>i</em>-th row of the matrix in logistic xtrain.npy is the predictor for the response in the <em>i</em>-th entry of the vector in logistic ytrain.npy, and analogously for the two test set files. Please include these files in your submission so that we can run your code without downloading them again. <strong>Note: </strong>we didn’t discuss reading numpy data from files. To load the files, you can simply call xtrain = np.load(’xtrain.npy’) to read the data into the variable xtrain. xtrain will be a Numpy array.

Load the training data and use it to obtain estimates of <em>W </em>and <em>b </em>by minimizing the negative log-likelihood via gradient descent. <strong>Another note: </strong>you’ll have to play around with the learning rate and the number of steps. Two good ways to check if optimization is finding a good minimizer:

<ul>

 <li>Try printing the training data loss before and after optimization.</li>

 <li>Use the test data to validate your estimated parameters.</li>

</ul>

<ol start="3">

 <li><strong>Evaluating logistic regression on test data. </strong>Load the test data. What is the negative log-likelihood of your model on this test data? That is, what is the negative log-likelihood when you use your estimated parameters with the previously unseen test data?</li>

 <li><strong>Evaluating the estimated logistic parameters. </strong>The data was, in reality, generated with</li>

</ol>

<em>W </em>= (1<em>,</em>1<em>,</em>2<em>,</em>3<em>,</em>5<em>,</em>8)<em>,               b </em>= −1<em>.</em>

Write TensorFlow expressions to compute the squared error between your estimated parameters and their true values. Evaluate the error in recovering <em>W </em>and <em>b </em>separately. What are the squared errors? <strong>Note: </strong>you need only evaluate the error of your final estimates, not at every step.

<ol start="5">

 <li>For ease of grading, please make the variables from the above problems availablein a dictionary called results_logistic. The dictionary should have keys ’W’,</li>

</ol>

’Wsqerr’, ’b’, ’bsqerr’, ’log_lik_test’ , with respective values sess.run(x) where x ranges over the corresponding quantities. For example, if my squared error for <em>W </em>is stored in a TF variable called W_squared_error, then the key ’Wsqerr’ should have value sess.run(W_squared_error).

<ol start="6">

 <li><strong>Classification of normally distributed data. </strong><a href="http://www-personal.umich.edu/~klevin/teaching/Winter2019/STATS507/HW10_normal.zip"> </a>contains four Numpy .npy files that contain train and test data generated from <em>K </em>= 3 different classes. Each class <em>k </em>∈{1<em>,</em>2<em>,</em>3} has an associated mean <em>µ<sub>k </sub></em>∈ <em>R </em>and variance <em>σ<sub>k</sub></em><sup>2 </sup>∈ R, and all observations from a given class are i.i.d. N(<em>µ<sub>k</sub>,σ<sub>k</sub></em><sup>2</sup>). The four files are:

  <ul>

   <li>npy : contains a 500-vector whose entries are the independent variables (predictors) from the test set.</li>

   <li>npy : contains a 2000-vector whose entries are the independent variables (predictors) from the train set.</li>

   <li>npy : contains a 500-by-3 dimensional matrix whose rows are one-hot encodings of the class labels for the test set.</li>

   <li>npy : contains a 2000-by-3 dimensional matrix whose rows are one-hot encodings of the class labels for the train set.</li>

  </ul></li>

</ol>

The <em>i</em>-th entry of the vector in normal_xtrain.npy is the observed random variable from class with label given by the <em>i</em>-th row of the matrix in normal_ytrain.npy, and analogously for the two test set files. Please include these files in your submission so that we can run your code without downloading them again.

Load the training data and use it to obtain estimates of the vector of class means <em>µ </em>= (<em>µ</em><sub>0</sub><em>,µ</em><sub>1</sub><em>,µ</em><sub>2</sub>) and variances) by minimizing the cross-entropy between the estimated normals and the one-hot encodings of the class labels (as we did in our softmax regression example in class). Please name the corresponding variables mu and sigma2. This time, instead of using gradient descent, use Adagrad, supplied by TensorFlow as the function tf.train.AdagradOptimizer. Adagrad is a <em>stochastic gradient descent algorithm</em>, popular in machine learning. You can call this just like the gradient descent optimizer we used in class—just supply a learning rate. Documentation for the TF implementation of Adagrad can be found here: <a href="https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer">https://www. </a><a href="https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer">tensorflow.org/api_docs/python/tf/train/AdagradOptimizer</a><a href="https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer">.</a> See <a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent">https:// </a><a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent">en.wikipedia.org/wiki/Stochastic_gradient_descent</a> for more information about stochastic gradient descent and the Adagrad algorithm.

<strong>Note: </strong>you’ll no longer be able to use the built-in logit cross-entropy that we used for training our models in lecture. Your cross-entropy for one observation should now look something like −<sup>P</sup><em><sub>k </sub>y<sub>k</sub></em><sup>0 </sup>log<em>p<sub>k</sub>, </em>where <em>y</em><sup>0 </sup>is the one-hot encoded vector and <em>p </em>is the vector whose <em>k</em>-th entry is the (estimated) probability of the <em>k</em>-th observation given its class. <strong>Another note: </strong>do not include any estimation of the mixing coefficients (i.e., the class priors) in your model. You only need to estimate three means and three variances, because we are building a <em>discriminative </em>model in this problem.

<ol start="7">

 <li><strong>Evaluating loss on test data. </strong>Load the test data. What is the cross-entropy of your model on this test data? That is, what is the cross-entropy when you use your estimated parameters with the previously unseen test data?</li>

 <li><strong>Evaluating parameter estimation on test data. </strong>The true parameter values for the three classes were <em>µ</em><sub>0 </sub>= −1<em>,σ</em><sub>0</sub><sup>2 </sup>= 0<em>.</em>5 <em>µ</em><sub>1 </sub>= 0<em>,σ</em><sub>1</sub><sup>2 </sup>= 1 <em>µ</em><sub>2 </sub>= 3<em>,σ</em><sub>2</sub><sup>2 </sup>= 1<em>.</em>5<em>.</em></li>

</ol>

Write a TensorFlow expression to compute the total squared error (i.e., summed over the six parameters) between your estimates and their true values. What is the squared error? <strong>Note: </strong>you need only evaluate the error of your final estimates, not at every step.

<ol start="9">

 <li><strong>Evaluating classification error on test data. </strong>Write and evaluate a TensorFlow expression that computes the classification error of your estimated model averaged over the test data.</li>

 <li>Again, for ease of grading, define a dictionary called results_class, with keys</li>

</ol>

’mu’, ’sigma2’, ’crossent_test’, ’class_error’ with keys corresponding to the evaluation (again using sess.run) of your estimate of <em>µ</em>, <em>σ</em><sup>2</sup>, the cross-entropy on the test set, and the classification error from the previous problem.

<h1>3         Building a Complicated Model</h1>

The TensorFlow documentation includes tutorials on building a number of more complicated neural models in TensorFlow: <a href="https://www.tensorflow.org/tutorials/">https://www.tensorflow.org/tutorials/</a><a href="https://www.tensorflow.org/tutorials/">.</a> In the left side panel, choose any one tutorial from under one of the headings “ML at production scale”, “Generative models”, “Images” or “Sequences” and follow it. Some of the tutorials include instructions along the lines of “We didn’t discuss this trick, try adding it!”. You do not need to do any of these additional steps (though you will certainly learn something if you do!). <strong>Warning: </strong>some of the tutorials require large amounts of training data. If this is the case, please do not include the training data in your submission! Instead, include a line of code to download the data from wherever it is stored. Also, some of the tutorials require especially long training time, (e.g., the neural models) so budget your time accordingly!

Your submission for this problem should be a <em>separate </em>jupyter notebook called tutorial.ipynb (no need to include your uniqname), which includes code to load the training and test data, build and train a model, and evaluate that model on test data. That is, the code in tutorial.ipynb should perform all the training and testing steps performed in the tutorial, but without having to be run from the command line. Depending on which model you choose, training may take a long time if you use the preset number of training steps, so be sure to include a variable called nsteps that controls the number of training steps, and set it to be something moderately small for your submission.

<strong>Note: </strong>it will not be enough to simply copy the tutorial’s python code into your jupyter notebook, since the demo code supplied in the tutorials is meant to be run from the command line.

<strong>Another note: </strong>If it was not clear, you are, for this problem and this problem only, permitted to copy-paste code from the TensorFlow tutorials as much as you like without penalty.

<strong>One more note: </strong>Please make sure that in both tutorial.ipynb and your main submission notebook uniqname.hw10.ipynb you do not set any training times to be excessively long. You are free to set the number of training steps as you like for running on your own machine, but please set these parameters to something more reasonable in your submission so that we do not need to wait too long when running your notebook. Aim to set the number of training steps so that we can run each of your submitted notebooks less than a minute.

<h1>4         Running Models on Google Cloud Platform</h1>

In this problem, you’ll get a bit of experience running TensorFlow jobs on Google Cloud Platform (GCP), Google’s cloud computing service. Google has provided us with a grant, which will provide each of you with free compute time on GCP.

<strong>Important: </strong>this problem is <strong>very hard</strong>. It involves a sequence of fairly complicated operations in GCP. As such, I do not expect every student to complete it. Don’t worry about that. Unless you’ve done a lot of programming in the past, this problem is likely your first foray into learning a new tool largely from scratch instead of having my lectures to guide you. The ability to do this is a crucial one for any data scientist, so consider this a learning opportunity (and a sort of miniature final exam). Start early, read the documentation carefully, and come to office hours if you’re having trouble.

Good luck, and have fun!

The first thing you should do is claim your share of the grant money by visiting this link: <a href="https://google.secure.force.com/GCPEDU?cid=VYZbhLIwytS0UVxuWxYyRYgNVxPMOf37oBx0hRmx71pfjlHwvVnxlUdVjBD6l5XA">https://google.secure.force.com/GCPEDU?cid=VYZbhLIwytS0UVxuWxYyRYgNVxPMOf37oBx0hRmx</a>

You will need to supply your name and your UMich email. Please use the email address associated to your unique name (i.e., <a href="/cdn-cgi/l/email-protection" class="__cf_email__" data-cfemail="2f5a41465e414e424a6f5a42464c47014a4b5a">[email protected]</a>), so that we can easily determine which account belongs to which student. Once you have submitted this form, you will receive a confirmation email through which you can claim your compute credits. These credits are valid on GCP until they expire in January 2020. Any credits left over after completing this homework are yours to use as you wish. Make sure that you claim your credits while signed in under your University of Michigan email, rather than a personal gmail account so that your project is correctly associated with your UMich email. If you accidentally claim the credits under a different address, add your unique name email as an owner.

Once you have claimed your credits, you should create a project, which will serve as a repository for your work on this problem. You should name your project uniqname-stats507w19, where uniqname is your unique name in all lower-case letters. Your project’s billing should be automatically linked to your credits, but you can verify this fact in the billing section dashboard in the GCP browser console. Please add both me (UMID klevin) and your GSI Roger Fan (UMID rogerfan) as owners. You can do this in the IAM tab of the IAM &amp; admin dashboard by clicking “Add” near the top of the page, and listing our UMich emails and specifying our Roles as Project → Owner.

<strong>Note: </strong>this problem is comparatively complicated, and involves a lot of moving parts. At the end of this problem (several pages below), I have included a list of all the files that should be included in your submission for this problem, as well as a list of what should be on your GCP project upon submission.

<strong>Important: </strong>after the deadline (May 2nd at 10:00am) you <strong>should not </strong>edit your GCP project in any way until you receive a grade for the assignment in canvas. If your project indicates that any files or running processes have been altered after the deadline by a user other than klevin or rogerfan, we will assume this to be an instance of editing your assignment after the deadline, and you will receive a penalty.

<ol>

 <li>Follow the tutorial at <a href="https://cloud.google.com/ml-engine/docs/distributed-tensorflow-mnist-cloud-datalab">https://cloud.google.com/ml-engine/docs/distributed</a><a href="https://cloud.google.com/ml-engine/docs/distributed-tensorflow-mnist-cloud-datalab">tensorflow-mnist-cloud-datalab</a><a href="https://cloud.google.com/ml-engine/docs/distributed-tensorflow-mnist-cloud-datalab">,</a> which will walk you through the process of training a CNN similar to the one we saw in class, but this time using resources on GCP instead of your own machine. This tutorial will also have you set up a DataLab notebook, which is Google’s version of a Jupyter notebook, in which you can interactively draw your own digits and pass them to your neural net for classification. <strong>Important: </strong>the tutorial will tell you to tear your nodes and storage down at the end. Do not do that. Leave everything running so that we can verify that you set things up correctly. It should only cost a few dollars to leave the datalab server and storage buckets running, but if you wish to conserve your credits, you can tear everything down and go through the tutorial again on the evening of May 1st or the (early!) morning of May 2nd.</li>

 <li>Let us return to the classifier that you trained above on the normally-distributeddata. In this and the next several subproblems, we will take an adaptation of that model and upload it to GCP where it will serve as a prediction node similar to the one you built in the tutorial above. Train the same classifier on the same training data, but this time, save the resulting trained model in a directory called normal_trained. You’ll want to use the saved_model.simple_save function. Refer to the GCP documentation at <a href="https://cloud.google.com/ml-engine/docs/deploying-models">https://cloud.google.com/ml-engine/docs/deploying-models</a><a href="https://cloud.google.com/ml-engine/docs/deploying-models">,</a></li>

</ol>

and the documentation on the tf.saved_model.simple_save function, here: <a href="https://www.tensorflow.org/programmers_guide/saved_model#save_and_restore_models">https:</a>

<a href="https://www.tensorflow.org/programmers_guide/saved_model#save_and_restore_models">//www.tensorflow.org/programmers_guide/saved_model#save_and_restore_models </a>Please include a copy of this model directory in your submission. <strong>Hint: </strong>a stumbling block in this problem is figuring out what to supply as the inputs and outputs arguments to the simple_save function. Your arguments should look something like inputs = {’x’:x}, outputs = {’prediction’:prediction}.

<ol start="3">

 <li>Let’s upload that model to GCP. First, we need somewhere to put your model. Youalready set up a bucket in the tutorial, but let’s build a separate one. Create a new bucket called uniqname-stats507w19-hw10-normal, where uniqname is your uniqname. You should be able to do this by making minor changes to the commands you ran in the tutorial, or by following the instructions at</li>

</ol>

<a href="https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine#creating_a_cloud_storage_bucket">https://cloud.google.com/solutions/running-distributed-tensorflow-on</a><a href="https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine#creating_a_cloud_storage_bucket">compute-engine#creating_a_cloud_storage_bucket</a><a href="https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine#creating_a_cloud_storage_bucket">.</a> Now, we need to upload your saved model to this bucket. There are several ways to do this, but the easiest is to follow the instructions at <a href="https://cloud.google.com/storage/docs/uploading-objects">https://cloud.google.com/storage/docs/ </a><a href="https://cloud.google.com/storage/docs/uploading-objects">uploading-objects</a> and upload your model through the GUI. <strong>Optional challenge (worth no extra points, just bragging rights): </strong>Instead of using the GUI, download and install the Google Cloud SDK, available at <a href="https://cloud.google.com/sdk/">https://cloud. </a><a href="https://cloud.google.com/sdk/">google.com/sdk/</a> and use the gsutil command line tool to upload your model to a storage bucket.

<ol start="4">

 <li>Now we need to create a <em>version </em>of your model. Versions are how the GCP machine learning tools organize different instances of the same model (e.g., the same model trained on two different data sets). To do this, follow the instructions located at <a href="https://cloud.google.com/ml-engine/docs/deploying-models#creating_a_model_version">https://cloud.google.com/ml-engine/docs/deploying-models#creating_a_mo</a>del_ <a href="https://cloud.google.com/ml-engine/docs/deploying-models#creating_a_model_version">version</a><a href="https://cloud.google.com/ml-engine/docs/deploying-models#creating_a_model_version">,</a> which will ask you to

  <ul>

   <li>Upload a SavedModel directory (which you just did)</li>

   <li>Create a Cloud ML Engine model resource</li>

   <li>Create a Cloud ML Engine version resource (this specifies where your model is stored, among other information)</li>

   <li>Enable the appropriate permissions on your account.</li>

  </ul></li>

</ol>

Please name your model stats507w19_hw10_normal (note the underscores here as opposed to the hyphens in the bucket name and note that this model name should not include your uniqname; see the documentation for the gcloud ml-engine versions command for how to delete versions, if need be). <strong>Important: </strong>there are a number of pitfalls that you may encounter here, which I want to warn you about: A good way to check that your model resource and version are set up correctly is to run the command gcloud ml-engine versions describe “your_version_name” –model “your_model_name”. The resulting output should include a line reading state: READY. You may notice that the Python version for the model appears as, say, pythonVersion: ’2.7’, even though you used, say, Python 3.6. This should not be a problem, but you <strong>should </strong>make sure that the runtimeVersion is set correctly. If the line runtimeVersion: ’1.0’ is appearing when you describe your version, you are likely headed for a bug. You can prevent this bug by adding the flag

–runtime-version 1.6 to your gcloud ml-engine versions create command, and making sure that you are running TensorFlow version 1.6 on your local machine (i.e., the machine where you’re running Jupyter). Running version 1.7 locally while running 1.6 on GCP also seems to work fine.

<ol start="5">

 <li>Create a .json file corresponding to a single prediction instance on the input observation <em>x </em>= 4. Name this .json file hw10.json, and please include a copy of it in your submission. <strong>Hint: </strong>you will likely find it easiest to use nano/vim/emacs to edit edit the .json file from the tutorial (GCP Cloud Shell has versions of all three of these editors). Doing this will allow you to edit a copy of the .json file directly in the GCP shell instead of going through the trouble of repeatedly downloading and uploading files. Being proficient with a shell-based text editor is also, generally speaking, a good skill for a data scientist to have.</li>

 <li>Okay, it’s time to make a prediction. Follow the instructions at <a href="https://cloud.google.com/ml-engine/docs/online-predict#requesting_predictions">https://cloud. </a><a href="https://cloud.google.com/ml-engine/docs/online-predict#requesting_predictions">com/ml-engine/docs/online-predict#requesting_predictions</a> to submit the observation in your .json file to your running model. Your model will make a prediction, and print the output of the model to the screen. Please include a copy-paste of the command you ran to request this prediction as well as the resulting output. Which cluster does your model think <em>x </em>= 4 came from? <strong>Hint: </strong>if you are getting errors about dimensions being wrong, make sure that your instance has the correct dimension expected by your model. <strong>Second hint: </strong>if you are encountering an error along the lines of Error during model execution:</li>

</ol>

AbortionError(code=StatusCode.INVALID_ARGUMENT, details=”NodeDef mentions attr ’output_type’, this is an indication that there is a mismatch between the version of TensorFlow that you used to create your model and the one that you are running on GCP. See the discussion of gcloud ml-engine versions create above.

That’s all of it! Great work! Here is a list of all files that should be included for this problem in your submission, as well as a list of what processes or resources should be left running in your GCP project:

<ul>

 <li>You should leave the datalab notebook and its supporting resources (i.e., the prediction node and storage bucket) from the GCP ML tutorial running in your GCP project.</li>

 <li>Include in your submission a copy of the saved model directory constructed from your classifier. You should also have a copy of this directory in a storage bucket on GCP.</li>

 <li>Leave a storage bucket running on GCP containing your uploaded model directory. This storage bucket should contain a model with a single version.</li>

 <li>Include in your submission a .json file representing a single observation. You need not include a copy of this file in a storage bucket on GCP; it will be stored by default in your GCP home directory if you created it in a text editor in the GCP shell.</li>

 <li>Include in your jupyter notebook a copy-paste of the command you ran to request your model’s prediction on the .json file, and please include the output that was printed to the screen in response to that prediction request. <strong>Note: </strong>Please make sure that the cell(s) that you copy-paste into is/are set to be Raw NBconvert cell(s), so that your commands display as code but are not run as code by Jupyter.</li>

</ul>