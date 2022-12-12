# Reinforcement-Learning-Agent for Qwox

This repository contains a Pettingzoo Environment for the dice Game "Qwox" :-) as well as multiple trained and algorithmic Agents
for this Environment.

## Local installation

`conda env update --file environment.yml --prune`

`brew install cmake openmpi`


## Manual Playing

`python3 src/manual_testing/play_by_hand.py`

Then put in the index you want to choose and press Enter.

It is also possible to change the agent you want to play against. By changing the number in `get_trained_agent(env, 103)`
There are  alot of pretrained agents available. The most important ones are:

* 103 Best Performing Agent, which is the 3rd Generation of trained Agents against other DQN based agents
* 107 1st Generation of a trained Agent
* 106 Agent which was trained with a limited Observation Space (only its own Card)

Also other Agents that use an Algorithmic Implementation are available with RandomPolicy(), LongPlayingPolicy() and LowestValueTaker()

## Run Evaluations against other Agents

`python3 src/manual_testing/play_without_training.py` 

Lets you run battles between different Agents of your choice. The results are the automatically saved into test-log.csv
The Agents can be set the same ways as in `Manual Playing` above


## Environment Documentation

This is a Pettingszoo Environment for the dice game Qwox 

| Attribute          | Description                        |
|--------------------|------------------------------------|
| Actions            | Discrete                           |
| Parallel API       | No                                 |
| Manual Control     | No (but custom implementation)     |
| Agents             | `agents= ['player_1', 'player_2']` |
| Agents             | 2-*                                |
| Action Shape       | Discrete(52)                       |
| Action Values      | Discrete(52)                       |
| Observation Shape  | (count_of_agents + 1, 5,12)        |
| Action Mask Shape  | (5,11)                             |
| Observation Values | int8                               |



### Observation Space
The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.

#### Observation
Observation contains (agent_count+1) 2-Dimensional arrays with shape (5,12).

It includes the game-cards of the other players as well as the dice values and round part. The shape is (player_count + 1, 5,12).
5,12 is one game-card state, whereas the first axis represent the other players cards (plus one for dice values and state-information). 
The last channel represents the dice values.  Important is that each player sees its own card on index 0 and the other ones after that.

#### Legal Actions Mask
The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. 
The `action_mask` is a binary vector with shape (5, 11) where each index of the vector represents whether the action is 
legal or not. 

### Action Space

Actions Space has the size 55. This has the same amount of values as the action mask with shape (5, 11). The input is
used as the index where a number on the game card should be crossed. Indexes are set like this:
```
[[ 0  1  2  3  4  5  6  7  8  9 10]
[11 12 13 14 15 16 17 18 19 20 21]
[22 23 24 25 26 27 28 29 30 31 32]
[33 34 35 36 37 38 39 40 41 42 43]
[44 45 46 47 48 49 50 51 52 53 54]]
```

* 0 - 43 are numbers that can be crossed
* 44 - 47 are passes (-5 reward)
* 48 - 54 does nothing and is used to skip any action

### Rewards

Rewards in Qwox are calculated by counting the crossed values per row and sum them up with the formula:

    1 -> 1 Point
    2 -> 3 Points
    3 -> 6 Points
    ...
    12 -> 78 Points

This can be calculated by doing 1+2+3, if the user has 3 numbers crossed

Any used pass gives a reward of -5



