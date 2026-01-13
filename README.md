# CaptureTheFlag

## Notes and findings

* TODO explain why QMIX
* Had a problem with the environment observation (could only use Box with new API)
* Had a problem with forward (Problem 1: https://discuss.ray.io/t/confusion-migrating-to-new-api/21783)
* TODO maybe also evaluate ray tune against algo.train()?


1. Problem: Agents learned to stick to the top, change reward structure:
![img.png](img.png)
In general agents sticked to areas where punishment was low (e.g. on the edge of the map just moving left right to minimize punishment), therefore I had to punish them if delta was zero
2. I then implemented different policies for each team because it seamed with the shared policies the agents agreed to a "stalemate" to minimize loss for each other because no flag was capture. Also maybe corners are local optimum?
3. Introduce more reward for closing in on the enemy flag, higher entropy to encourage exploration, more reward for capturing the flag.
4. Time penalty too large relative to movement reward: -0.01 per step. The corner behavior makes perfect sense: the agents learned that doing nothing loses less reward than random wandering!
5. Add penalty for hitting the edge of the playing field.

Example with corner:
![img_1.png](img_1.png)