# Revisiting DeepFool: generalization and improvement

Official PyTorch implementation of " Revisiting DeepFool: generalization and improvement "

<p align="center">
<img src="https://github.com/alirezaabdollahpour/SuperDeepFool/blob/main/images/oghab.png" alt="Demo" height="350" width="750"/>
</p>

# Abstract
Deep neural networks have been known to be vulnerable
to adversarial examples, which are inputs that are modified
slightly to fool the network into making incorrect predictions.
This has led to a significant amount of research on evaluating the robustness of these networks against such perturbations. One particularly important robustness metric is the
robustness to minimal ℓ2 adversarial perturbations. However, existing methods for evaluating this robustness metric
are either computationally expensive or not very accurate. In
this paper, we introduce a new family of adversarial attacks
that strike a balance between effectiveness and computational efficiency. Our proposed attacks are generalizations
of the well-known DeepFool (DF) attack, while they remain
simple to understand and implement. We demonstrate that
our attacks outperform existing methods in terms of both
effectiveness and computational efficiency. Our proposed
attacks are also suitable for evaluating the robustness of
large models and can be used to perform adversarial training (AT) to achieve state-of-the-art robustness to minimal ℓ2
adversarial perturbations.

## Illustration of SuperDeepFool
<p align="center">
<img src="https://github.com/alirezaabdollahpour/SuperDeepFool/blob/main/images/illus.png" alt="illus" height="420" width="720"/>
</p>

# Repository Structure

```
└── attacks             # Source code for attacks
    ├── attack          # Attack class
    ├── DeepFool        # DeepFool
    ├── SuperDeepFool   # SuperDeepFool
└── curvature           # source code of curvature
    └── curvature       # curvature analysis
└── utils             # Source code for utils
  ├── utils            # utils function

└── Dockerfile        # Docker scripts
└── main.py           # main scripts
```

# Running in Docker <img src="https://github.com/alirezaabdollahpour/SuperDeepFool/blob/main/images/docker.png" alt="docker" style="float:right; margin-right: 2px; width: 150px;">

<!DOCTYPE html>
<html>
  <body>
    <h1>Dockerfile Documentation</h1>
    <p>This Dockerfile is used to build a Docker image that runs a Python application. The application is stored in the /app directory within the container, and the Docker image is based on the official Python 3.9 slim-buster image.</p>
    <h2>Instructions</h2>
    <p>The following instructions are executed in order when the Dockerfile is built:</p>
    <ol>
      <li>The working directory is set to /app using the <code>WORKDIR</code> command.</li>
      <li>All files in the current directory are copied to the /app directory in the container using the <code>COPY</code> command.</li>
      <li>The packages specified in the <code>requirements.txt</code> file are installed using the <code>RUN</code> command and the <code>pip install</code> command. The <code>--no-cache-dir</code> flag is used to prevent the package manager from caching package installation files.</li>
      <li>Port 80 is exposed using the <code>EXPOSE</code> command. This tells Docker that the container will listen on port 80 when it is run.</li>
      <li>An environment variable named <code>NAME</code> is defined using the <code>ENV</code> command. This variable can be accessed by the application running in the container.</li>
      <li>The <code>CMD</code> command is used to specify the command that should be run when the container is launched. In this case, the <code>python app.py</code> command is run to start the Python application.</li>
    </ol>
    <h2>Building the Docker Image</h2>
    <p>To build the Docker image, navigate to the directory containing the Dockerfile and run the following command:</p>
    <pre><code>docker build -t &lt;image-name&gt; .</code></pre>
    <p>Replace &lt;image-name&gt; with the desired name for the Docker image.</p>
    <h2>Running the Docker Container</h2>
    <p>To run the Docker container, use the following command:</p>
    <pre><code>docker run --gpus all -it  NAME=&lt;name&gt; &lt;image-name&gt;</code></pre>
    <p>Replace &lt;name&gt; with the desired value for the <code>NAME</code> environment variable, and replace &lt;image-name&gt; with the name of the Docker image you built in the previous step. This command will run the container and enable access to all available GPUs on the host machine.</p>
  </body>
</html>

<!DOCTYPE html>
<html>
    <h1>Running with python</h1>
    <p>To use this Python project, follow the instructions below:</p>
    <ol>
      <li>Clone the repository</li>
      <li>Open the terminal or command prompt and navigate to the cloned repository</li>
      <li>Run the command "python main.py" to start the project</li>
      <li>All results will be stored in the "results" folder</li>
    </ol>
  </body>
</html>


## Results
Below, we provide the results with SDF. We test our algorithms on deep convolutional neural network architectures trained on MNIST, CIFAR-10, and
ImageNet datasets. For the evaluation, we use all the
MNIST test dataset, while for CIFAR-10 and ImageNet we
use 1000 samples randomly chosen from their corresponding
test datasets. For MNIST, we use a robust model called [IBP](https://github.com/huanzhang12/CROWN-IBP). For CIFAR-10, we use three models: an adversarially trained [PreActResNet-18](https://openreview.net/forum?id=Azh9QBQ4tR7), a regularly
trained Wide ResNet 28-10 (WRN-28 − 10) and
LeNet. These models are obtainable via the [RobustBench](https://robustbench.github.io/). On ImageNet, we test the attacks on two
ResNet-50 (RN-50) models: one regularly trained and one
ℓ2 adversarially trained, obtainable through the [robustness library](https://github.com/MadryLab/robustness).

### Comparison with DeepFool in terms of mean and the median of &#8467;<sub>2</sub>-norm of perturbations
| Attack | Mean-&#8467;<sub>2</sub> | Median-&#8467;<sub>2</sub> | Grads |
|---|:---:|:---:|:---:|
| DF | 0.17 |0.15 | 14 |
| SDF | 0.11 | 0.10 | 32|

### Comparison with DeepFool in terms of orthogonality
| Attack | Lenet | ResNet-18 | WRN-28-10 |
|---|:---:|:---:|:---:|
| DF | 0.89 |0.14 | 0.21 |
| SDF | 0.92 | 0.72 | 0.80|

### ImageNet ResNet-50
| Attack | FR | Median-&#8467;<sub>2</sub> | Grads |
|---|:---:|:---:|:---:|
| ALMA | 100 |0.10 | 100 |
| DDN | 99.9 |0.17 | 1000 |
| FAB | 99.3 | 0.10 | 900|
| FMN | 99.3 | 0.10 | 1000|
| C&W | 100 | 0.21 | 82667|
| SDF | 100 | 0.09 | 37|

## Adversarial Training 
we evaluate the performance of a model
adversarially trained using SDF against minimum-norm attacks and [AutoAttack](https://arxiv.org/abs/2003.01690). Our experiments provide valuable
insights into the effectiveness of adversarial training with
SDF and sheds light on its potential applications in building
more robust models. 
We adversarially train a WRN-28-10 on CIFAR-10. Similar to the procedure followed [Madry et al,](https://scholar.google.ch/citations?view_op=view_citation&hl=en&user=SupjsEUAAAAJ&citation_for_view=SupjsEUAAAAJ:IWHjjKOFINEC), we restrict ℓ2-norms
of perturbation to 2.6 and set the maximum number of iterations for SDF to 6. We train the model on clean examples
for the first 200 epochs, and we then fine-tune it with SDF
generated adversarial examples for 60 more epochs. Our
model reaches a test accuracy of 90.8%.

## Curvature analysis
For part of our experiments, we analyzed the curvature of the models. 
We've simplified the code from [Efficient Training of
Low-Curvature Neural Networks](https://arxiv.org/pdf/2206.07144.pdf) paper and modified it to be easier to use with [RobustBench](https://robustbench.github.io/). Please refer to [CURE](https://openaccess.thecvf.com/content_CVPR_2019/papers/Moosavi-Dezfooli_Robustness_via_Curvature_Regularization_and_Vice_Versa_CVPR_2019_paper.pdf) and [Efficient Training of
Low-Curvature Neural Networks](https://arxiv.org/pdf/2206.07144.pdf) for a more detailed analysis.
For use our code please cd to curvature folder and run this command :
```
python curvature.py
```

## Implementations
In terms of implementation, we use [Pytorch](https://github.com/pytorch/pytorch) . [Foolbox](https://github.com/bethgelab/foolbox) and [Torchattcks](https://github.com/Harry24k/adversarial-attacks-pytorch) libraries are used to implement adversarial
attacks. Our code follows a similar pattern to [Torchattcks](https://github.com/Harry24k/adversarial-attacks-pytorch), and we plan to integrate it with [Torchattcks](https://github.com/Harry24k/adversarial-attacks-pytorch) in the near future. This will enable us to leverage the benefits of [Torchattcks](https://github.com/Harry24k/adversarial-attacks-pytorch) and enhance the capabilities of our code for adversarial attacks in deep learning models.


