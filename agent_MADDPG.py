import torch
import random
import numpy as np
from collections import deque # store memory
from game_fish import SnakeGameAI, INITIAL_FISH_NUM
from model import Linear_QNet, QTrainer
from helper import plot
import torch.nn as nn
import torch.optim as optim
from variables_n_utils import sort_by_distance, get_sign

MAX_MEMORY = 10_000
BATCH_SIZE = 100
LR_ACTOR = 0.001
LR_CRITIC = 0.002
GAMMA = 0.95
TAU = 0.01
NUM_ACTIONS = 4
PLOT_LEARNING = False

'''
- State
Shark (danger) direction
Other agent's direction

- Action
move up/down/left/right

Agent를 train할때 필요한 함수들을 모음
실제 agent는 game.py에 구현되어 있음

action:
up down left right

state:
- hidden state: absolute coordinate of the fish
- observed state: relative coordinates of other fish / shark
'''

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
        
class Agent:
    def __init__(self, state_size=0, output_size=4):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate 0~1
        self.memory = deque(maxlen=MAX_MEMORY)  # 꽉차면 popleft()
        if state_size == 0:
            state_size = 2 + 2 * (INITIAL_FISH_NUM - 1)
        self.model = Linear_QNet(state_size, 256, output_size)  # 2(shark) + 2*(#fish - 1) input, 4 action outputs
        self.trainer = QTrainer(self.model, lr = LR_ACTOR, gamma = self.gamma)

    def reset(self):
        pass

    def get_state(self, game, fish_to_update): # game 으로부터 agent의 state를 계산
        fish = game.fish_list[fish_to_update]
        shark = game.shark
        shark_x = shark.x - fish.x
        shark_y = shark.y - fish.y

        # sort nearby fishes by distance and excludes the fish (me)
        sorted_fish_list = sort_by_distance(game.fish_list, fish)

        # other fish's relative vector
        other_fish_state = []
        for i in range(INITIAL_FISH_NUM-1): # 각 input의 위치가 list가 줄어듦에 따라 변할 수 있다. 그러나 각각의 물고기에 대한 가중치는 대칭적으로 동일해야 하기 때문에 이렇게 처리해도 괜찮아야 한다 (즉 물고기마다 특별하지 않다)
            if (len(sorted_fish_list)<= i): #if current fish is dead => 없는거나 다름없게 state를 주자: 거리가 0 이도록 주면 된다. 그러면 물고기가 해당 물고기에게 다가가기 위해 이동할 필요가 없어지기 때문이다
                other_fish_state.append(0)
                other_fish_state.append(0)
                continue

            friend_fish = sorted_fish_list[i]
            other_fish_state.append(get_sign(friend_fish.x - fish.x))
            other_fish_state.append(get_sign(friend_fish.y - fish.y))

        state = [
            # Danger: 현재 상어의 방향 부호만 줌
            get_sign(shark_x),
            get_sign(shark_y),

            # Food location
            # game.food.x < game.fish.x, # food left
            # game.food.x > game.fish.x, # food right
            # game.food.y < game.fish.y, # food up
            # game.food.y > game.fish.y, # food down

        ] + other_fish_state
        return np.array(state, dtype = int) # float로 바꾸려면 dtype=float


    # experience replay
    def remember(self, state, action, reward, next_state, done): # done = game over state
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: # random sample
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        # states, actions, rewards, next_states, dones
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self,state, action, reward, next_state, done): # train for one game step
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff btw exploration / exploitation
        self.epsilon = 40 - self.n_games # parameter # 원래는 80 - self.n_games
        final_move = [0,0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # execute forward function in the model
            move = torch.argmax(prediction).item() # convert to only one number = item
            final_move[move] = 1

        return final_move
    
class MADDPGAgent(Agent):
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actors = [Actor(state_dim, hidden_dim, action_dim) for _ in range(num_agents)]
        self.actor_targets = [Actor(state_dim, hidden_dim, action_dim) for _ in range(num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=LR_ACTOR) for actor in self.actors]

        self.critic = Critic(state_dim * num_agents + action_dim * num_agents, hidden_dim, 1)
        self.critic_target = Critic(state_dim * num_agents + action_dim * num_agents, hidden_dim, 1)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.memory = deque(maxlen=MAX_MEMORY)
        self.gamma = GAMMA
        self.tau = TAU
        self.n_games = 0

        for actor, target in zip(self.actors, self.actor_targets):
            target.load_state_dict(actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, agent_index):
        state = torch.tensor(state, dtype=torch.float)
        action_probs = self.actors[agent_index](state)
        action = np.random.choice(self.action_dim, p=action_probs.detach().numpy())
        action_onehot = np.zeros(self.action_dim)
        action_onehot[action] = 1
        return action_onehot

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        mini_batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in mini_batch:
            state = torch.tensor(state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.float)
            reward = torch.tensor(reward, dtype=torch.float)
            next_state = torch.tensor(next_state, dtype=torch.float)
            done = torch.tensor(done, dtype=torch.float)

            # Update critic
            next_actions=[]
            for i in range(self.num_agents):
                action_probs = self.actors[i](state)
                act = np.random.choice(self.action_dim, p=action_probs[i].detach().numpy())
                action_onehot = np.zeros(self.action_dim)
                action_onehot[act] = 1
                action_onehot=torch.tensor(action_onehot, dtype=torch.float)
                next_actions.append(action_onehot)
            next_actions = torch.cat(next_actions, dim=0)
            next_state_action = torch.cat((next_state.view(self.state_dim*self.num_agents), next_actions), dim=0)
            target_q = reward + (1 - done) * self.gamma * self.critic_target(next_state_action)
            expected_q = self.critic(torch.cat((state.view(self.state_dim*self.num_agents), action.view(self.num_agents*self.action_dim)), dim=0))
            critic_loss = nn.MSELoss()(expected_q, target_q.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update actors
            for i in range(self.num_agents):
                self.actor_optimizers[i].zero_grad()
                self.actors[i](state[i])
                current_actions=action.clone()
                act = np.random.choice(self.action_dim, p=action_probs[i].detach().numpy())
                action_onehot = np.zeros(self.action_dim)
                action_onehot[act] = 1
                action_onehot=torch.tensor(action_onehot, dtype=torch.float)
                current_actions[i]=action_onehot
                
                actor_loss = -self.critic(torch.cat((state.view(self.state_dim*self.num_agents), current_actions.view(self.action_dim*self.num_agents)), dim=0)).mean()
                actor_loss.backward()
                self.actor_optimizers[i].step()

            # Soft update target networks
            self.soft_update(self.critic, self.critic_target, self.tau)
            for actor, target in zip(self.actors, self.actor_targets):
                self.soft_update(actor, target, self.tau)




def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = -999 # best score
    agent = MADDPGAgent(INITIAL_FISH_NUM,2 + 2*(INITIAL_FISH_NUM-1),4)
    game = SnakeGameAI()
    iters=0
    while True:
        iters+=1
        '''
        현재 매커니즘: NN model 하나를 모든 agent가 공유해서 사용함
        매 loop 마다 모든 물고기의 state와 action을 구하고
        game loop를 한번 돌린 후 
        모든 물고기에 대해 model parameter를 업데이트 함 => 1개를 골라서 하나의 물고기에 대해서만 업데이트 해도 될까? (speed issue)
        
        '''
        state_olds = []
        final_moves = []
        for i in range(len(game.fish_list)):
            # if (i >= len(game.fish_list)):
            #     break # go to next loop if fish do not exist
            # get old state
            cur_state_old = agent.get_state(game, i)
            state_olds.append(cur_state_old)

            # get move
            final_moves.append(agent.act(cur_state_old,i))
        reward, done, score = game.play_step(final_moves)
        state_news = []
        for i in range(len(game.fish_list)):
            cur_state_new = agent.get_state(game, i)
            state_news.append(cur_state_new)
        
        agent.remember(state_olds, final_moves, reward, state_news, done)
        if iters%500==0:
            agent.train()
        if done:
            game.reset()
            agent.n_games += 1
            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record: ', record)

            # plotting
            if PLOT_LEARNING:
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
    


