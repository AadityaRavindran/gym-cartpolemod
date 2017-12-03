This repository contains a PIP package which is a modified version of the 
CartPole-v0 OpenAI environment which includes cart and pole friction and also
adds a new 'do nothing' action.


## Installation

Install [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
sudo pip install -e .
```

## Usage
Python usage:
```
import gym
import gym_cartpolemod

env = gym.make('CartPoleMod-v0')
```
Examples:
Versions go from v0 through v6 for different noise scenarios
```
cd example_files
python deepQNetwork.py v1
```

## The Environment

Some parameters for the cart-pole system:
- mass of the cart = 1.0
- mass of the pole = 0.1
- length of the pole = 0.5 
- magnitude of the force = 10.0
- friction at the cart = 5e-4
- friction at the pole = 2e-6

Noise cases(v0 to v6):
- v0: Noise free
- v1: 5%  Uniform Actuator noise
- v2: 10% Uniform Actuator noise
- v3: 5%  Uniform Sensor noise
- v4: 10% Uniform Sensor noise
- v5: 5%  Gaussian Sensor noise
- v6: 10% Gaussian Sensor noise

## The team
- Aaditya Ravindran
- Apoorva Sonavani
- Rohith Krishna Gopi

This was created as a part of Prof. Jennie Si's class on Artificial Neural Computation (Fall 2017) at Arizona State University

Special thanks to [MartinThoma](https://github.com/MartinThoma/banana-gym), [Kevin Frans](https://github.com/kvfrans/openai-cartpole), and [keon](https://keon.io/deep-q-learning/)
