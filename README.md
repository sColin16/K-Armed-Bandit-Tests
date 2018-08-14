# K-armed Bandit Tests

A simple script to explore how different variables (specifically epsilon,
initial Q-table values and alpha) affect the performance of a k-armed
bandit.

## Usage

1. Create a virtual environment<br/>
```python3 -m virtualenv venv```
2. Install required packages<br/>
```pip3 install -r requirements.txt```
3. Edit the 'Experiment Variable' lists<br/>
```
EPSILONS = [0, 0.1, 0.2]
INITIALS = [0, 1, 5]
ALPHAS = [None, 0.1, 0.2]
```
4. Run `python3 main.py`

## Pre-made results

Some interesting results using the script were run and the graphs saved.
The default values for each variable were `Epsilon = 0.1`, `Initial = 0`, 
and `Alpha = None` (for sample-average update method). Unless otherwise
noted, 2000 trials were performed for 1000 episodes, with a stationary
problem.

Generalizing these results too much should be avoided. Although certain
values may seem "optimal," the k-armed bandit is a simple, stateless 
problem, whose performance depends on a wide variety of factors, not all
of which were thoroughly considered (e.g. number of arms, variance in 
rewards, etc.). Although the patterns observed through these experiments
may be a good starting place, every problem will be different and
should be treated as such.


### Changing Epsilon

* [epsilon-reward.png](results/epsilon-reward.png) - Average reward over episodes
* [epsilon-optimal.png](results/epsilon-optimal.png) - Chance bandit found optimal policy after so many episodes

These results suggest that higher epsilon values generally increase the
probability that the bandit will find the optimal policy in early episodes.
However, too high of an epsilon (e.g. `Epsilon = 1`) slightly decreases the
probability the bandit will find the optimal policy. Furthermore, lower
epsilon values (e.g. 0.05) keep pace with higher epsilon values (e.g. 0.5)
with regards to average reward in early episodes, and then outperform 
them in the long run.


### Changing initial Q-Table values

* [initial-reward.png](results/initial-reward.png) - Average reward receivied by bandit for different initial Q-Table values
* [initial-optimal.png](results/initial-optimal.png) - Chance bandit found optimal policy after so many episodes

These results suggest that more optimisitc values (i.e. more positive
initial Q-table values) greatly improve both the probability that the
bandit will learn the optimal policy and the average reward the bandit
receives in early episodes. However, in the long run this change does
not significantly impact the chance of learning an optimal policy or
the average reward.

## Non Stationary Problems

Due to the fact that many reinforcement learning problems are 
non-stationary (i.e. the enviornment changes) a variety of tests with
non-stationary problems were performed. The same default values were
used as for the stationary problems.

Note that for these problems, the graphs have more noise, suggesting it is
more difficult to train agents on non-stationary problems

### Changing Alpha

* [alpha-reward.png](results/alpha-reward.png) - Average reward
* [alpha-optimal.png](results/alpha-optimal.png) - Chance of optimal policy

These results demonstrate the need for a constant step-size when the
problem is non-stationary. Although the sample-average update method
has a high probability of finding the optimal policy for the first 1000
or so iterations, its performance drops significantly, and is overtaken
by every constant step-size bandit by 10000 iterations. An alpha of 0.1
seems to be optimal for the parameters selected, which adds another
hyperparameter to tune. The average reward graph is not valuable, as 
it simply reflects that every model was capable of finding more valuable
rewards as the episode number increased.

### Changing Epsilon
(These tests were done with `Alpha = 0.1` for 4000 episodes)

* [epsilon-ns-reward.png](results/epsilon-ns-reward.png)
* [epsilon-ns-optimal.png](results/epsilon-ns-optimal.png)

These results are strikingly similar to the stationary problem. As before,
higher epsilons (below 1.0), tend to increase the probability that the
agent finds the optimal policy. And, as before, lower epsilons tend to
outperform higher epsilons in the long run (other than 0)

## Your results

Find any interesting results? Share them with me by submitting a pull request!

## Roadmap

Potential changes in the future

1. Provide ways to change epsilon, whether a step or continuous function 
2. Provide more control over the step size (alpha)
3. Calculate the average total reward
4. Change the variance in lever means
5. Change the variance in the non-stationary problem
6. Optimize. I feel there are lots of optimization to be made, especially
in checking if the max arg changed, to avoid recalculation

## References

This script and the experiment are largely inspired by the book 
"Reinforcement Learning: An Introduction" by Richard Sutton and 
Andrew Barto. A pdf version can be found online 
[here](http://incompleteideas.net/book/bookdraft2017nov5.pdf). 
