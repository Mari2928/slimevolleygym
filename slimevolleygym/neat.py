"""
Simple NEAT policy trained using NEAT-Python.
"""

import numpy as np
import pickle

from neat.config import Config
from neat.genes import DefaultNodeGene
from neat.genome import DefaultGenome
from neat.population import Population
from neat.reproduction import DefaultReproduction
from neat.species import DefaultSpeciesSet
from neat.stagnation import DefaultStagnation
import curses
import itertools
import math
import time
from neat.nn import FeedForwardNetwork
from visualize_network import*
import neat


def makeSlimeNeatPolicy(filename):
  model = NeatModel() 
  model.load_model(filename)
  return model

class NeatModel:
  ''' simple feedforward neat model '''
  def __init__(self):
    self.nGameInput = 8 # 8 states for agent
    self.nGameOutput = 3 # 3 buttons (forward, backward, jump)

    self.nOutput = self.nGameOutput
    self.nInput = self.nGameInput+self.nOutput

    self.c = Config(
    DefaultGenome,
    DefaultReproduction,
    DefaultSpeciesSet,
    DefaultStagnation,
    "slimevolleygym/simple.conf",
    )
    self.p = Population(self.c)
    self.net = None
    self.stats = neat.StatisticsReporter()
    self.p.add_reporter(self.stats)
    # self.p.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    # self.p.add_reporter(neat.Checkpointer(100, 900))

    # store current inputs and outputs
    self.inputState = np.zeros(self.nInput)
    self.outputState = np.zeros(self.nOutput)

  def make_env(self, seed=-1, render_mode=False):
    self.render_mode = render_mode
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode)

  def _setInputState(self, obs):
    # obs is: (op is opponent). obs is also from perspective of the agent (x values negated for other agent)
    [x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy] = obs
    self.inputState[0:self.nGameInput] = np.array([x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy])
    self.inputState[self.nGameInput:] = self.outputState

  def _getAction(self):
    forward = 0
    backward = 0
    jump = 0
    if (self.outputState[0] > 0.75):
      forward = 1
    if (self.outputState[1] > 0.75):
      backward = 1
    if (self.outputState[2] > 0.75):
      jump = 1
    return [forward, backward, jump]
  
  def predict(self, obs):
     """ take obs, update rnn state, return action """
     self._setInputState(obs)
     self.outputState = self.net.activate(self.inputState)
     return self._getAction()
  
  def set_model_params(self, genome):
    self.net = FeedForwardNetwork.create(genome, self.c)
  
  def load_model(self, filename):
    with open(filename, 'rb') as f:
      best_genome = pickle.load(f)
    print('loading file %s' % (filename))
    self.set_model_params(best_genome)

    # visualize the best network
    # node_names = create_node_names(best_genome, self.c)
    # dot = visualize_network_with_activations(best_genome, self.c, node_names)
    # dot.render(f'marimari/bestnet_with_act', view=True)