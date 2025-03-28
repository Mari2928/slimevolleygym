# Trains a NEAT agent against Baseline agent that uses RNN.
# Not optimized for speed, and just uses a single CPU (mainly for simplicity).

import os
import numpy as np
import gym
import pickle
import slimevolleygym
from slimevolleygym.neat import NeatModel
from slimevolleygym import multiagent_rollout as rollout
from eval_agents import evaluate_multiagent
from visualize_network import*
import visualize
import matplotlib.pyplot as plt

# Settings
random_seed = 612
total_tournaments = 1 

# Log results
logdir = "results/neat_bl_p500_multiacts0.5_op2_gen100" 
if not os.path.exists(logdir):
  os.makedirs(logdir)

# Create two instances of a feed forward policy we may need later.
policy_left = slimevolleygym.BaselinePolicy()
policy_right = NeatModel()

# create the gym environment, and seed it
env = gym.make("SlimeVolley-v0")
env.seed(random_seed)
np.random.seed(random_seed)

class ScoreCompute():
    
  def __init__(self):
    self.generation = 0
    self.history = []
    self.gen_best = None
    self.solved = False

  def eval_genome(self, genome, config):
    policy_right.set_model_params(genome)
    fitnesses = []
    for _ in range(10):
      score, length, fitness = rollout(env, policy_right, policy_left, render_mode=False)
      fitnesses.append(fitness)

    return np.mean(fitnesses)

  def eval_genomes(self, genomes, config):
    for id, genome in genomes:
      fitness = self.eval_genome(genome, config)
      genome.fitness = fitness
        
    # print stats of the best genome for every generations
    best_genomes = policy_right.stats.best_unique_genomes(1)
    best_genome = None
    for g in best_genomes:
        policy_right.set_model_params(g) 
        best_genome = g
    scores = []
    lengths = []
    fitnesses = []
    if self.generation % 10 == 0:   
      history = evaluate_multiagent(env, policy_right, policy_left)
      score_ave = np.round(np.mean(history), 3)
      print("neat scored", score_ave, "±", np.round(np.std(history), 3), "vs baseline", "over", "1000 trials.")
  
      if score_ave > 1.4:
        sc.solved = True

      # save the checkpoint genome
      model_filename = os.path.join(logdir, str(self.generation) +".pckl")
      f = open(model_filename, 'wb')
      pickle.dump(best_genome, f)
        
    else:
      for k in range(100):
        score, step, fitness = rollout(env, policy_right, policy_left)
        scores.append(score)
        lengths.append(step)  
        fitnesses.append(fitness)     
      print("generation: ",self.generation, 
            "mean_score: ", np.mean(scores), 
            "score_std: ", np.round(np.std(scores), 3), 
            "mean_step", np.mean(lengths),
            "mean_fitness:", np.round(np.mean(fitnesses), 3),
            "solved:", sc.solved) 
     
    self.generation += 1

if __name__ == '__main__':
  sc = ScoreCompute()
  sc.gen_best = policy_right.p.run(sc.eval_genomes, 100)
  # save visualized stats
  visualize.plot_stats(policy_right.stats, ylog=False, view=True, filename=os.path.join(logdir, "fitness.svg")))
  plt.grid()
  plt.legend(loc='best')    
  plt.savefig(os.path.join(logdir, "scores.svg"))
  plt.close()

  # save the best network
  model_filename = os.path.join(logdir, "neat_winner"+".pckl")
  f = open(model_filename, 'wb')
  pickle.dump(sc.gen_best, f)
  f.close()

  # visualize the best network
  node_names = create_node_names(sc.gen_best, policy_right.c)
  dot = visualize_network_with_activations(sc.gen_best, policy_right.c, node_names)
  dot.render(f'{logdir}/bestnet_with_act', view=True)
