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

ppo_v3 --> good example for reward hacking
agents where going to the flag and stopped at a break even where going forward
brought more reward than going away to the flag
Then they started to wiggle back and forth. Sometimes one agent went to the flag while one was staying behind
reward hacking.
```
if delta_distance > 0:
    # reward += delta_distance * (1 / (dist + 0.1))
    reward += delta_distance
else:
    reward += delta_distance * 0.2
```

Then I tried no penalty for moving away from enemy flag to reduce agents fleeing to corners out of fear from penalties.