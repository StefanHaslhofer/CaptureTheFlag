# CaptureTheFlag

## Notes and findings

* TODO explain why QMIX
* Had a problem with the environment observation (could only use Box with new API)
* Had a problem with forward (Problem 1: https://discuss.ray.io/t/confusion-migrating-to-new-api/21783)
* TODO maybe also evaluate ray tune against algo.train()?


1. Problem: Agents learned to stick to the top, change reward structure:
![img.png](img.png)
In general agents sticked to areas where punishment was low (e.g. on the edge of the map just moving left right to minimize punishment), therefore I had to punish them if delta was zero