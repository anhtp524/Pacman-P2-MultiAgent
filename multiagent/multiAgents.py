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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if (successorGameState.isWin()):
            return 10000


        score = successorGameState.getScore()
        foodArray = newFood.asList()
        cost = [manhattanDistance(newPos,food) for food in foodArray]
        min_food = min(cost)
        score = score + 1.0/min_food


        for ghost in newGhostStates:
            ghostpos = ghost.getPosition()
            ghostDist = util.manhattanDistance(ghostpos, newPos)
            if (manhattanDistance(newPos,ghostpos)) < 2:
                return -10000
        return score
        'return successorGameState.getScore()'

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        return self.MinimaxSearch(gameState, 1, 0)

    def MinimaxSearch(self, gameState, currentDepth, agentIndex):
        "terminal check"
        if currentDepth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        "minimax algorithm"
        legalMoves = [action for action in gameState.getLegalActions(
            agentIndex) if action != 'Stop']

        # Cap nhat do sau tiep theo
        nextIndex = agentIndex + 1
        nextDepth = currentDepth
        if nextIndex >= gameState.getNumAgents():
            nextIndex = 0
            nextDepth += 1

        # Chon phuong an di tot nhat
        results = [self.MinimaxSearch(gameState.generateSuccessor(agentIndex, action),
                                      nextDepth, nextIndex) for action in legalMoves]
        if agentIndex == 0 and currentDepth == 1:  # pacman di chuyen lan dau tien
            bestMove = max(results)
            bestIndices = [index for index in range(
                len(results)) if results[index] == bestMove]
            # chon random
            chosenIndex = random.choice(bestIndices)
            return legalMoves[chosenIndex]

        if agentIndex == 0:
            bestMove = max(results)
            return bestMove
        else:
            bestMove = min(results)
            return bestMove


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num_of_agents = gameState.getNumAgents()
        # khoi tao gia tri alpha va beta bang vo cung ~
        value, action = self.alpha_value(
            gameState, float('-inf'), float('inf'), 0, self.depth)
        return action

    # gia tri alpha khi toi luot pacman
    def alpha_value(self, state, alpha, beta, agentNumber, depth):

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state), 'none'

        v = float('-inf')
        actions = state.getLegalActions(agentNumber)
        bestAction = actions[0]

        for action in actions:
            previous_v = v
            successorGameState = state.generateSuccessor(agentNumber, action)
            # kiem tra cac nodes la va ktra xem da ket thuc chua
            if depth == 0 or successorGameState.isWin() or successorGameState.isLose():
                v = max(v, self.evaluationFunction(successorGameState))
            else:
                v = max(v, self.beta_value(successorGameState,
                                           alpha, beta, agentNumber+1, depth))
            # kiem tra dieu kien cat tia alpha-beta pruning
            if v > beta:
                return v, action
            alpha = max(alpha, v)

            # luu lai Action tot nhat
            if v != previous_v:
                bestAction = action
        return v, bestAction

    # gia tri beta
    def beta_value(self, state, alpha, beta, agentNumber, depth):

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state), 'none'

        v = float('inf')
        actions = state.getLegalActions(agentNumber)
        flag = False
        for action in actions:

            successorGameState = state.generateSuccessor(agentNumber, action)
            if depth == 0 or successorGameState.isWin() or successorGameState.isLose():
                v = min(v, self.evaluationFunction(successorGameState))
            elif agentNumber == (state.getNumAgents() - 1):

                # flag dung de tranh truong hop giam do sau cho cung 1 level nhieu hon 1 lan
                if flag == False:
                    depth = depth - 1
                    flag = True

                # kiem tra da vuot qua level cuoi hay chua
                if depth == 0:
                    v = min(v, self.evaluationFunction(successorGameState))
                else:
                    v = min(v, self.alpha_value(
                        successorGameState, alpha, beta, 0, depth)[0])

            else:
                v = min(v, self.beta_value(successorGameState,
                                           alpha, beta, agentNumber+1, depth))
            if v < alpha:  # kiem tra dieu kien cat tia alpha beta prunning
                return v
            beta = min(beta, v)

        return v

    # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        num_of_agents = gameState.getNumAgents()
        depth1 = self.depth * num_of_agents

        self.getAction1(gameState, depth1, num_of_agents)
        return self.action1
        util.raiseNotDefined()

    def getAction1(self, gameState, depth1, num_of_agents):
        maxvalues = list()
        chancevalues = list()
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if depth1 > 0:
            if depth1 % num_of_agents == 0:
                agentNumber = 0

            else:
                agentNumber = num_of_agents-(depth1 % num_of_agents)

            actions = gameState.getLegalActions(agentNumber)
            for action in actions:
                successorGameState = gameState.generateSuccessor(
                    agentNumber, action)

                if agentNumber == 0:
                    maxvalues.append(
                        (self.getAction1(successorGameState, depth1-1, num_of_agents), action))
                    maximum = max(maxvalues)
                    self.value_max = maximum[0]
                    self.action1 = maximum[1]

                else:
                    chancevalues.append(
                        (self.getAction1(successorGameState, depth1-1, num_of_agents), action))
                    avg = 0.0
                    for i in chancevalues:
                        avg += chancevalues[chancevalues.index(i)][0]
                    # Tra ve trung binh cac lan ghost di chuyen
                    avg /= len(chancevalues)
                    self.value_avg = avg

            if agentNumber == 0:
                return self.value_max
            else:
                return self.value_avg

        else:
            # nodes la tra ve evaluated score
            return self.evaluationFunction(gameState)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    ghostStates = map(lambda x: (abs(x.getPosition()[
                      0] - newPos[0]) + abs(x.getPosition()[1] - newPos[1]), x.scaredTimer), newGhostStates)
    evaluation = currentGameState.getScore()

    for state in ghostStates:
        if state[0] == 0 and state[1] == 0:
            return -999999
        if state[0] == 1 and state[1] == 0:
            evaluation -= 1000
        if state[0] < state[1] and state[1] > 0:
            evaluation += 50

    totFoodDist = 0
    for foodPos in newFood:
        totFoodDist -= (abs(foodPos[0] - newPos[0]) +
                        abs(foodPos[1] - newPos[1]))**0.55

    evaluation -= len(newFood) * 12

    if len(newFood) > 0:
        evaluation += totFoodDist / len(newFood)

    else:
        evaluation += 999999

    return evaluation
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
