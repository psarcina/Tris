#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:37:34 2020

@author: pasq
"""

import numpy as np

class Agent():
    def __init__(self, t=0, action_t=[0], state=np.zeros((3,3)), actions=np.ones((3,3)), tree={0:np.zeros(2)}):
        self.t = t
        self.action_t = action_t.copy()
        self.state = state.copy()
        self.actions = actions.copy()
        self.tree = tree.copy()
        
    
    def move(self, eps):
        if eps == "human":
            print("Please player %d select row and column in the format row, col")
            r, c = input().split(",")
            r, c = int(r), int(c)
            pos = r*3+c
                    
            #check valid move:
            while not(bool(self.actions[r,c])):
                print("Not a legal move. Please try again.")
                r, c = input().split(",")
                r, c = int(r), int(c)
            
                
        elif eps == "random":
            pos = np.array(range(9))[self.actions.flatten().astype(bool)]
            pos = pos[np.floor(np.random.rand()*len(pos)).astype(int)]
            r = pos//3
            c = pos%3
            
        else:
            r = eps//3
            c = eps%3
            pos = eps
            
            
#        print("Selected action is:" + str(pos))
        return r, c, pos
        
    
    def evaluate(self, state, i):
        vert_vict = any(state.sum(axis=0) == 3)
        hori_vict = any(state.sum(axis=1) == 3)
        diag_vict = (state.diagonal().sum() == 3)
        antd_vict = (np.fliplr(state).diagonal().sum() == 3)
        if vert_vict or hori_vict or diag_vict or antd_vict:
            return (-1)**i

    
    def update(self, eps):
        r, c, act = self.move(eps)
        print("The selected action is: %s" % act)
        self.state[r,c] = 1
        self.actions[r,c] = 0
        self.t += 1
        self.action_t.append(act + self.action_t[-1]*9 + 1)
        
        
    def play_game(self, how="random", tree_size=9):
        #Initialization
        msg = "Yeah, victory"
        reward = None
        i = 0
        
        #Game loop
        while ((reward == None)):
            pl = (-1)**(self.t)
            if self.actions.sum() == 0:
                msg = "This is a draw"
                reward = 0
                break
            
            self.state = self.state*(-1)
#            print("iteration %s with state values:" % pl)
#            print(self.state)
            self.update("random")
            reward = self.evaluate(self.state, i)
            i += 1
            
#        print(msg)
#        print(self.state)
        return reward, self.action_t, pl
        


def backpropagation(tree, r, acts, tree_size=9):
    for a in acts:
        tree[a] = tree.get(a, 0) + np.array([r,1])
        
#    tree[0] = tree.get(0,0) + np.array([r,1])
        
    return tree
    

def expansion(action_t, tree, tree_size=9):
    k = action_t[-1]
    for i in range(tree_size):
        tree[k*tree_size+i+1] = tree.get(k*tree_size+i+1,0)+np.array([0,0])
    
    return tree


def selection(agent, tree_size=9):
    u_t = []        
    for i in range(agent.action_t[-1]*tree_size+1, agent.action_t[-1]*tree_size+10):
        r,n = agent.tree[i]
        n_tot = agent.tree[agent.action_t[-1]][1]
        u = r/(1+n) + np.sqrt(n_tot)/(1+n)
        u_t.append(u)
        
    act = np.argmax(np.array(u_t))
    agent.update(act)
    return agent


  
def mcts(agent, num_iter=100):
    i = 0
    while i < num_iter:
        print(i)
        if agent.tree[agent.action_t[-1]][1] == 0:
#            new_agent = Agent(agent.t, agent.action_t.copy(), agent.state.copy(), agent.actions.copy(), agent.tree)
#            reward, acts, _ = new_agent.play_game()
#            agent.tree = backpropagation(new_agent.tree, reward, acts)
            reward, acts, _ = agent.play_game()
            agent.tree = backpropagation(agent.tree, reward, acts)
            tree = agent.tree.copy()
            agent = Agent()
            agent.tree = tree.copy()
            print("Rollout and Backprop")
            i += 1
#            print(agent.t)
#            print(agent.tree)
        else:
            agent.tree = expansion(agent.action_t, agent.tree)
            agent = selection(agent)
            print("Expansion and Selection")
#            print(agent.t)
#            print(agent.tree)
            
    
    print(agent.t)
    print(agent.action_t)
    print(agent.state)
    print(agent.actions)
    for n,k in enumerate(agent.tree.keys()):
        print(str(k)+": "+str(agent.tree[k]))
        if n > 20: break
    
    
    decision = []
    for i in range(agent.action_t[-1]*9+1, agent.action_t[-1]*9+10):
        print(agent.tree[i][0]/agent.tree[i][1])
        decision.append(agent.tree[i][0]/agent.tree[i][1])
        
    np.argmax(np.array(decision))
