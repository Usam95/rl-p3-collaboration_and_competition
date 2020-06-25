
from torch import device, cuda

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
UPDATE_EVERY = 20       # how often to update the network

LR_ACTOR  = 1e-3		# learning rate of the actor 
LR_CRITIC = 1e-3		# learning rate of the critic 
WEIGHT_DECAY = 0.0		# L2 weight decay
N_LEARN_UPDATES = 2	    # number of learning updates
SEED = 42
STOP_NOISE_AFTER_EP=300

N_TIME_STEPS  = 10      # every n time step do update
FC1_UNITS = 256			# Number of nodes in first hidden layer of Actor and Critic
FC2_UNITS = 256			# Number of nodes in second hidden layer of Actor and Critic

# Parameters used to control the training
PRINT_EVERY = 200        # How often to print average score during training
SOLVED_SCORE = 0.5      # Average score needed to solve the RL task
CONSEC_EPISODES = 100   # How many episodes at least SOLVED_SCORE must be achieved to solve the RL task
device = device("cuda:0" if cuda.is_available() else "cpu")
