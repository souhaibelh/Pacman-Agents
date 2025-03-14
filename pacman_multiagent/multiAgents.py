# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random, util, math

from game import Agent, Directions

from util import manhattan_distance

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        successor_game_state = current_game_state.generatePacmanSuccessor(action)
        new_pos = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood().asList(key=True)
        new_ghost_states = successor_game_state.getGhostStates()
        new_scared_times = [ghost.scaredTimer for ghost in new_ghost_states]
        new_capsules = successor_game_state.getCapsules()
        score = successor_game_state.getScore()

        if new_food:
            min_food_distance = min([manhattan_distance(new_pos, food) for food in new_food])
            score = score + (10 / (min_food_distance + 1))

        if new_capsules:
            min_capsule_distance = min([manhattan_distance(new_pos, capsule) for capsule in new_capsules])

            nearby_ghost = any(manhattan_distance(new_pos, ghost.getPosition()) < 10 for ghost in new_ghost_states)
            
            if nearby_ghost and min_capsule_distance < 10:
                score = score + (15 / (min_capsule_distance + 1))
            elif min_capsule_distance < 5:
                score = score + (10 / (min_capsule_distance + 1))

        for ghost, scared_time in zip(new_ghost_states, new_scared_times):
            ghost_distance = manhattan_distance(new_pos, ghost.getPosition()) 

            if scared_time == 0:
                if ghost_distance < 4:
                    score = score - (100 / (ghost_distance + 1))
                else:
                    score = score - (5 / (ghost_distance + 1))
            else:
                score = score + ((20 / (ghost_distance + 1)) * scared_time)

        return score

def score_evaluation_function(current_game_state):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search game
      (not reflex game).
    """
    return current_game_state.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search game.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'better', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        initial_depth = 0
        agent_index = 0
        bestScore, bestAction = self.get_action_helper(initial_depth, game_state, agent_index)

        return bestAction

    def get_action_helper(self, depth, game_state, agent_index):
      if depth == self.depth or game_state.isWin() or game_state.isLose():
        return self.evaluationFunction(game_state), None

      if agent_index == 0:
        best_score = float("-inf")
        best_action = None
        for action in game_state.getLegalActions():
          next_game_state = game_state.generatePacmanSuccessor(action)
          next_agent_index = 0 if agent_index + 1 == game_state.getNumAgents() else agent_index + 1
          next_score = self.get_action_helper(depth + 1, next_game_state, next_agent_index)[0]
          if next_score > best_score:
            best_score = next_score
            best_action = action
        return best_score, best_action
      else:
        best_score = float("inf")
        best_action = None
        for action in game_state.getLegalActions(agent_index):
          next_game_state = game_state.generateSuccessor(agent_index, action)
          next_agent_index = 0 if agent_index + 1 == game_state.getNumAgents() else agent_index + 1
          next_score = self.get_action_helper(depth + 1 if next_agent_index == 0 else depth, next_game_state, next_agent_index)[0]
          if next_score < best_score:
            best_score = next_score
            best_action = action
        return best_score, best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        initial_depth = 0
        agent_index = 0
        alpha = float("-inf")
        beta = float("inf")
        bestScore, bestAction = self.get_action_helper(initial_depth, game_state, agent_index, alpha, beta)

        return bestAction
        util.raise_not_defined()

    def get_action_helper(self, depth, game_state, agent_index, alpha, beta):
      if depth == self.depth or game_state.isWin() or game_state.isLose():
        return self.evaluationFunction(game_state), None

      if agent_index == 0:
        best_score = float("-inf")
        best_action = None
        for action in game_state.getLegalActions():
          next_game_state = game_state.generatePacmanSuccessor(action)
          next_agent_index = 0 if agent_index + 1 == game_state.getNumAgents() else agent_index + 1
          next_score = self.get_action_helper(depth + 1, next_game_state, next_agent_index, alpha, beta)[0]
          if next_score > best_score:
            best_score = next_score
            best_action = action
          if best_score > beta:
            return best_score, best_action  
          alpha = max(alpha, best_score)
        return best_score, best_action
      else:
        best_score = float("inf")
        best_action = None
        for action in game_state.getLegalActions(agent_index):
          next_game_state = game_state.generateSuccessor(agent_index, action)
          next_agent_index = 0 if agent_index + 1 == game_state.getNumAgents() else agent_index + 1
          next_score = self.get_action_helper(depth + 1 if next_agent_index == 0 else depth, next_game_state, next_agent_index, alpha, beta)[0]
          if next_score < best_score:
            best_score = next_score
            best_action = action
          if best_score < alpha:
            return best_score, best_action
          beta = min(beta, best_score)   
        return best_score, best_action        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Votre agent Expectimax (question 4)
    """

    def get_action(self, game_state):
        """
          Retourne l'action basée sur l'algorithme Expectimax en utilisant self.depth et self.evaluationFunction
        """
        initial_depth = 0
        agent_index = 0
        best_score, best_action = self.get_action_helper(initial_depth, game_state, agent_index)
        return best_action

    def get_action_helper(self, depth, game_state, agent_index):
        if depth == self.depth or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state), None

        if agent_index == 0:
            best_score = float("-inf")
            best_action = None
            for action in game_state.getLegalActions(agent_index):
                next_game_state = game_state.generatePacmanSuccessor(action)
                next_agent_index = 0 if agent_index + 1 == game_state.getNumAgents() else agent_index + 1
                next_score = self.get_action_helper(depth, next_game_state, next_agent_index)[0]
                if next_score > best_score:
                    best_score = next_score
                    best_action = action
            return best_score, best_action
        else:
            actions = game_state.getLegalActions(agent_index)
            if not actions:
                return self.evaluationFunction(game_state), None
            total_score = 0
            for action in actions:
                next_game_state = game_state.generateSuccessor(agent_index, action)
                next_agent_index = 0 if agent_index + 1 == game_state.getNumAgents() else agent_index + 1
                next_score = self.get_action_helper(depth + 1 if next_agent_index == 0 else depth, next_game_state, next_agent_index)[0]
                total_score += next_score
            return total_score / len(actions), None

def aStar(gameState, goal, heuristic):
  """
  A* algorithm
  """
  start = gameState.getPacmanPosition()
  frontier = util.PriorityQueue()
  frontier.push((start, []), 0)
  explored = set()

  while not frontier.is_empty():
      current, path = frontier.pop()
      if current == goal:
          return path
      if current not in explored:
          explored.add(current)
          for next in gameState.getLegalActions():
             successor = gameState.generateSuccessor(0, next)
             nextPos = successor.getPacmanPosition()
             if nextPos not in explored:
                newPath = path + [next]
                newCost = len(newPath) + heuristic(nextPos, goal)
                frontier.push((nextPos, newPath), newCost)
  return []            

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    current_score = currentGameState.getScore()
    current_score -= currentGameState.getNumFood()
    pacman_pos = currentGameState.getPacmanPosition()
    ghost_states = currentGameState.getGhostStates()
    amount_scared_ghosts = num_scared_ghosts = sum(1 for ghost in ghost_states if ghost.scaredTimer > 0)
    current_score += 15 * amount_scared_ghosts
    ghost_positions = currentGameState.getGhostPositions()
    capsules = currentGameState.getCapsules()
    available_foods = currentGameState.getFood().asList(key=True)
    current_score -= len(capsules)

    if available_foods:
      min_food_distance = min(manhattan_distance(pacman_pos, food) for food in available_foods)
      current_score += (10 / (min_food_distance + 1))

    if capsules:
      min_capsule_distance = min(manhattan_distance(pacman_pos, capsule) for capsule in capsules)
      current_score += (15 / (min_capsule_distance + 1))  
    
    for ghost, position in zip(ghost_states, ghost_positions):
      distance_to_ghost = len(aStar(currentGameState, position, manhattan_distance))
      if ghost.scaredTimer == 0:
        if distance_to_ghost < 3:
          current_score -= 100 / (distance_to_ghost + 1)
      else:
        scared_reward = 200 / (distance_to_ghost + 1)
        
    return current_score
    util.raise_not_defined()

# Abbreviation
better = betterEvaluationFunction

