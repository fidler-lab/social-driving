# social-driving
Design multi-agent environments and simple reward functions such that social
driving behavior emerges


## Installation

We use MPI and OpenAI SpinningUp in our implementation of PPO. These need to
be setup before installing `sdriving`. Please follow the installation
instructions on [this page](https://spinningup.openai.com/en/latest/user/installation.html)
for proper setup.

Additionally pytorch needs to be installed. Follow the instructions given
[here](https://pytorch.org/get-started/locally/).

Finally install sdriving using the following instructions.

```
$ git clone https://github.com/fidler-lab/social-driving.git sdriving
$ cd sdriving
$ python setup.py install
```