import numpy as np

def state_norm(state):
      state[0] = state[0]/800
      state[1] = state[1]/800
      state[2] = state[2]/180
      state[3] = state[3]/180
      state[4] = state[4]/180
      state[5] = state[5]/180
      state[6] = state[6]/180
      state[7] = state[7]/180
      
      return state
