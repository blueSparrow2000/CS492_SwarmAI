import torch
import random
import numpy as np
from collections import deque # store memory
from game_fish import SwarmGameAI, INITIAL_FISH_NUM
from model import Linear_QNet, QTrainer
from helper import plot
import torch.nn as nn
import torch.optim as optim
from variables_n_utils import *
import math

MAX_MEMORY = 100000
BATCH_SIZE = 2000
LR_ACTOR = 0.001
LR_CRITIC = 0.001
GAMMA = 0.99
TAU = 0.01
NUM_ACTIONS = 4
PLOT_LEARNING = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
def gumbel_softmax(logits, tau=1, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()  # Sample from Gumbel distribution
    y = (logits + gumbels) / tau
    y = torch.softmax(y, dim=-1)
    if hard:
        # To make it a one-hot vector, take the argmax
        _, ind = y.max(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, ind, 1)
        y = y_hard - y.detach() + y
    return y
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
        self.model = Linear_QNet(state_size, 256, output_size).to(device)  # 2(shark) + 2*(#fish - 1) input, 4 action outputs
        self.trainer = QTrainer(self.model, lr = LR_ACTOR, gamma = self.gamma)

    def reset(self):
        pass

    def get_state(self, game, fish_to_update): # game 으로부터 agent의 state를 계산
        fish = game.fish_list[fish_to_update]
        shark = game.shark
        
        # 상어 거리 계산
        dx_shark = min(abs(shark.x - fish.x), WIDTH - abs(shark.x - fish.x))
        dy_shark = min(abs(shark.y - fish.y), HEIGHT - abs(shark.y - fish.y))
        shark_distance = math.sqrt(dx_shark ** 2 + dy_shark ** 2)
        
        shark_up = 0
        shark_down = 0
        shark_left = 0
        shark_right = 0
        
        # 물고기 시야 안에 상어가 있는 경우에만 상어의 방향을 알려줌
        if shark_distance <= FISH_VISION * BLOCK_SIZE:
            if shark.x > fish.x:
                if abs(shark.x - fish.x) > (WIDTH - abs(shark.x - fish.x)):
                    shark_left = 1
                else:
                    shark_right = 1
            elif shark.x < fish.x:
                if abs(shark.x - fish.x) > (WIDTH - abs(shark.x - fish.x)):
                    shark_right = 1
                else:
                    shark_left = 1
            if shark.y > fish.y:
                if abs(shark.y - fish.y) > (HEIGHT - abs(shark.y - fish.y)):
                    shark_down = 1
                else:
                    shark_up = 1
            elif shark.y < fish.y:
                if abs(shark.y - fish.y) > (HEIGHT - abs(shark.y - fish.y)):
                    shark_up = 1
                else:
                    shark_down = 1
        
        # sort nearby fishes by distance and excludes the fish (me)
        sorted_fish_list = sort_by_distance(game.fish_list, fish)
        
        up = 0
        down = 0
        left = 0
        right = 0

        for other_fish in  sorted_fish_list:
            if (other_fish.id == fish.id):
                continue
            
            dx = min(abs(other_fish.x - fish.x), WIDTH - abs(other_fish.x - fish.x))
            dy = min(abs(other_fish.y - fish.y), HEIGHT - abs(other_fish.y - fish.y))
            distance = math.sqrt(dx ** 2 + dy ** 2)

            # [상 하 좌 우]
            # 다른 물고기와 x 좌표 비교 후 오른쪽 +1 할지 왼쪽 +1 할지 결정
            if distance <= FISH_VISION * BLOCK_SIZE:
                if other_fish.x > fish.x:
                    if abs(other_fish.x - fish.x) > (WIDTH - abs(other_fish.x - fish.x)):
                        left += 1
                    else:
                        right += 1
                elif other_fish.x < fish.x:
                    if abs(other_fish.x - fish.x) > (WIDTH - abs(other_fish.x - fish.x)):
                        right += 1
                    else:
                        left += 1
                if other_fish.y > fish.y:
                    if abs(other_fish.y - fish.y) > (HEIGHT - abs(other_fish.y - fish.y)):
                        down += 1
                    else:
                        up += 1
                elif other_fish.y < fish.y:
                    if abs(other_fish.y - fish.y) > (HEIGHT - abs(other_fish.y - fish.y)):
                        up += 1
                    else:
                        down += 1

        # other fish's relative vector
        # other_fish_state = []
        # for i in range(INITIAL_FISH_NUM-1): # 각 input의 위치가 list가 줄어듦에 따라 변할 수 있다. 그러나 각각의 물고기에 대한 가중치는 대칭적으로 동일해야 하기 때문에 이렇게 처리해도 괜찮아야 한다 (즉 물고기마다 특별하지 않다)
        #     if (len(sorted_fish_list)<= i): #if current fish is dead => 없는거나 다름없게 state를 주자: 거리가 0 이도록 주면 된다. 그러면 물고기가 해당 물고기에게 다가가기 위해 이동할 필요가 없어지기 때문이다
        #         other_fish_state.append(0)
        #         other_fish_state.append(0)
        #         continue

        #     friend_fish = sorted_fish_list[i]
        #     other_fish_state.append(get_sign(friend_fish.x - fish.x))
        #     other_fish_state.append(get_sign(friend_fish.y - fish.y))

        state = [
            # Danger: 현재 상어의 방향 부호만 줌
            shark_up,
            shark_down,
            shark_left,
            shark_right,

            # 다른 물고기 수
            up,
            down,
            left,
            right

            # Food location
            # game.food.x < game.fish.x, # food left
            # game.food.x > game.fish.x, # food right
            # game.food.y < game.fish.y, # food up
            # game.food.y > game.fish.y, # food down

        ]
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
            state0 = torch.tensor(state, dtype=torch.float).to(device)
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

        self.actors = [Actor(state_dim, hidden_dim, action_dim).to(device) for _ in range(num_agents)]
        self.actor_targets = [Actor(state_dim, hidden_dim, action_dim).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=LR_ACTOR) for actor in self.actors]

        self.critic = [Critic(state_dim * num_agents + action_dim * num_agents, hidden_dim, 1).to(device) for _ in range(num_agents)]
        self.critic_target = [Critic(state_dim * num_agents + action_dim * num_agents, hidden_dim, 1).to(device) for _ in range(num_agents)]
        self.critic_optimizer =[optim.Adam(critic.parameters(), lr=LR_CRITIC) for critic in self.critic]

        self.memory = deque(maxlen=MAX_MEMORY)
        self.gamma = GAMMA
        self.tau = TAU
        self.n_games = 0

        for actor, target in zip(self.actors, self.actor_targets):
            target.load_state_dict(actor.state_dict())
        for cri, tar in zip(self.critic, self.critic_target):
            tar.load_state_dict(cri.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, agent_index):
        state = torch.tensor(state, dtype=torch.float).to(device)
        self.actors[agent_index].to(device)
        action_probs = self.actors[agent_index](state.unsqueeze(0))  # state를 GPU로 이동시킴
        action = torch.multinomial(action_probs.squeeze(0), 1).item()
        action_onehot = torch.zeros(self.action_dim, device=device)
        action_onehot[action] = 1
        return action_onehot.cpu().numpy()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        mini_sample = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # Convert to tensors and move to the appropriate device
        states = torch.tensor(np.array(states), dtype=torch.float).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float).unsqueeze(1).to(device)
        
        next_actions = []
        for i in range(self.num_agents):
            action_probs = self.actor_targets[i](next_states[:, i, :])
            actions_sampled = torch.multinomial(action_probs, 1)
            action_onehot = torch.zeros(BATCH_SIZE, self.action_dim, device=device)
            action_onehot.scatter_(1, actions_sampled, 1)
            next_actions.append(action_onehot)
        
        next_actions = torch.stack(next_actions, dim=1).to(device)
        next_state_action = torch.cat((next_states.view(BATCH_SIZE, -1), next_actions.view(BATCH_SIZE, -1)), dim=1).to(device)
        
        for i in range(self.num_agents):
            target_q = rewards + (1 - dones) * self.gamma * self.critic_target[i](next_state_action)
            combined_states_actions = torch.cat((states.view(BATCH_SIZE, -1), actions.view(BATCH_SIZE, -1)), dim=1).to(device)
            expected_q = self.critic[i](combined_states_actions)
            critic_loss = nn.MSELoss()(expected_q, target_q.detach())
            
            self.critic_optimizer[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[i].step()
        
        for i in range(self.num_agents):
            action_pred = self.actors[i](states[:, i, :])
            current_actions = actions.clone().to(device)
            current_actions[:, i, :] = action_pred
            combined_states_actions = torch.cat((states.view(BATCH_SIZE, -1), current_actions.view(BATCH_SIZE, -1)), dim=1).to(device)
            actor_loss = -self.critic[i](combined_states_actions).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
        
        for cri, tar in zip(self.critic, self.critic_target):
            self.soft_update(cri, tar, self.tau)
        
        for actor, target in zip(self.actors, self.actor_targets):
            self.soft_update(actor, target, self.tau)

BASELINE = '' #'' : no baseline # 'oneDir' : moves one direction #'random'

#olama
def train():
    plot_scores = []
    plot_mean_scores = []
    top_ten_scores = deque(maxlen=10)
    total_score = 0
    record = -999 # best score
    agent = MADDPGAgent(INITIAL_FISH_NUM, 8, 4)
    game = SwarmGameAI()
    iters=0
    while True:
        iters+=1
        '''
        현재 매커니즘: NN model 하나를 모든 agent가 공유해서 사용함
        매 loop 마다 모든 물고기의 state와 action을 구하고
        game loop를 한번 돌린 후 
        모든 물고기에 대해 model parameter를 업데이트 함 => 1개를 골라서 하나의 물고기에 대해서만 업데이트 해도 될까? (speed issue)
        
        '''
        ###################### Baseline #########################
        if BASELINE == 'random':
            final_moves = []
            for fish in game.fish_list:
                move_idx = random.randint(0, 3)
                my_move = [0,0,0,0]
                my_move[move_idx] = 1
                # get move
                final_moves.append(my_move)

            reward, done, score = game.play_step(final_moves)

            if done:
                game.reset()
                agent.n_games += 1
                if score > record:
                    record = score
                print('Game', agent.n_games, 'Score', score, 'Record: ', record)

                # plotting
                if PLOT_LEARNING:
                    plot_scores.append(score)
                    top_ten_scores.appendleft(score)
                    # total_score += score
                    # mean_score = total_score / agent.n_games
                    mean_score = sum(top_ten_scores) / len(top_ten_scores)
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)
        elif BASELINE == 'oneDir':
            final_moves = []
            for fish in game.fish_list:
                move_idx = fish.id % 4
                my_move = [0,0,0,0]
                my_move[move_idx] = 1
                # get move
                final_moves.append(my_move)

            reward, done, score = game.play_step(final_moves)

            if done:
                game.reset()
                agent.n_games += 1
                if score > record:
                    record = score
                print('Game', agent.n_games, 'Score', score, 'Record: ', record)

                # plotting
                if PLOT_LEARNING:
                    plot_scores.append(score)
                    top_ten_scores.appendleft(score)
                    # total_score += score
                    # mean_score = total_score / agent.n_games
                    mean_score = sum(top_ten_scores) / len(top_ten_scores)
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)
        ###################### Baseline #########################
        else:
        #################### ORIGINAL TRAINING ####################
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
            if iters%100==0:
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
                    top_ten_scores.appendleft(score)
                    # total_score += score
                    # mean_score = total_score / agent.n_games
                    mean_score = sum(top_ten_scores) / len(top_ten_scores)
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)
        ############### ORIGINAL TRAINING ####################

if __name__ == '__main__':
    train()
    


