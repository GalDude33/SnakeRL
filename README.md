# SnakeRL
Deep Reinforcement Learning implementation for the mighty game of Snake

### Agents Implemented:
[Linear Agent](policies/policy_linear.py)

[Basic DQN](policies/policy_DQN.py)

[Dueling DQN](policies/policy_DuelDQN.py)

[Convolutional DQN](policies/policy_ConvDQN.py)

[Convolutional Dueling DQN](policies/policy_ConvDuelDQN.py)

##### Important Paramters:
DoubleDQN - enables double DQN learning.

radius - control the radius of the agent field of view.

### Prioritized Experience Replay
Some experiences may be more important than others for our training, but might occur less frequently. Since the batch is sampled uniformly by default, these rich experiences that occur rarely have practically no chance to be selected. 

Changing sampling distribution by using a criterion to define the priority of each tuple of experience (Used the prediction mean absolute difference error) as a priority criterion to give the model more chance to learn from experiences on
which its’ predictin is bad, thus giving it a better chance to improve its’ predictions.

For usage:

Just change the base class for any of the above to [PriorBaseDQN](policies/policy_PriorityBaseDQN.py).

### State Representations
Tried multiple different implementations:

• Square(l∞ distance) of a certain radius around the snake’s head.

• Diamond(l1 distance) of a certain radius around the snake’s head.

• Circle(l2 distance) of a certain radius around the Snake’s head.

• Using sort of a “Radar”, which returned directions and distances to a certain amount of entities of each type from the snake’s head.

### Exploration-Exploitation trade-off: 
Used softmax sampling, with a temperature value that decays exponentially with a certain rate, in order to
explore more on early stage and then exploit the agent’s “knowledge” in later stage.

Also have option to use epsilon-greedy.



