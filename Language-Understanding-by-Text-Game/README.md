# Language Understanding for Text-based Games using Deep Reinforcement Learning  

https://arxiv.org/pdf/1506.08941.pdf

For single quest testing phase , success rate of our model is 100% with 200 epsisodes !

---

Deep model

---

![alt_text](https://github.com/wei-lin-liao/Deep-Reinforcement-Learning/blob/master/Language-Understanding-by-Text-Game/images/LSTM-DQN.PNG)

---

Testing outcome ( 100% success rate , only argmax Q(s,a) , not epsilon-greedy )

---

(1) ~ (5) represent different testing episodes.

"You are bored." is the quest. "You are in ...." is the descripition of location.

The goal of the quest "You are bored." is " watching TV in the living room " !!

It would learn how to select correct action to get more reward by itself, so the agent would go to living room and watch TV finally in this quest.

For example , in testing episode (1) , the agent reads the " descriptions " ( You are bored. You are in garden. ) from environment and takes action " go west " ( "You go west" is the record of actions. Agent would not read this sentence. ).

Next time step ( next line ) , the agent encounters the new situation ( You are bored. You are in the living room. ) and "watch TV".



![alt text](https://github.com/wei-lin-liao/Deep-Reinforcement-Learning/blob/master/Language-Understanding-by-Text-Game/images/Quest-01.png)

---

Room position of the environment 「Home World」

---

![alt text](https://github.com/wei-lin-liao/Deep-Reinforcement-Learning/blob/master/Language-Understanding-by-Text-Game/images/Home%20world.png)

---

Quest 01 episodes reward sum

---

![alt text](https://github.com/wei-lin-liao/Deep-Reinforcement-Learning/blob/master/Language-Understanding-by-Text-Game/images/Quest_01_reward_sum.png)

---

Quest 02 episodes reward sum

---

![alt text](https://github.com/wei-lin-liao/Deep-Reinforcement-Learning/blob/master/Language-Understanding-by-Text-Game/images/Quest_02_reward_sum.png)

---

Quest 03 episodes reward sum

---

![alt text](https://github.com/wei-lin-liao/Deep-Reinforcement-Learning/blob/master/Language-Understanding-by-Text-Game/images/Quest_03_reward_sum.png)

---

Quest 04 episodes reward sum

---

![alt text](https://github.com/wei-lin-liao/Deep-Reinforcement-Learning/blob/master/Language-Understanding-by-Text-Game/images/Quest_04_reward_sum.png)


