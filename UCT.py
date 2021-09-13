# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import math


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# MCTS Actions #
##########

class MCTSActions:
    """
    use MCTS to implement attack action
    """

    def __init__(self, agent, index, gameState):
        self.agent = agent
        self.index = index
        # used to store location of boundary
        self.boundary = []

        if self.agent.red:
            mid_line = (gameState.data.layout.width - 2) // 2
        else:
            mid_line = (gameState.data.layout.width - 2) // 2 + 1

        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(mid_line, i):
                self.boundary.append((mid_line, i))

    def getFeatures(self, gameState, action):
        features = util.Counter()
        return features

    def getWeights(self, gameState, action):
        weights = util.Counter()
        return weights

    def getUCB(self, V, N, n):
        if n == 0:
            return float('inf')
        else:
            ucb = V / n + 2 * math.sqrt(math.log(N) / n)
            return ucb

    def expand(self, gameState, decay):
        values = []
        actions = gameState.getLegalActions(self.index)
        # no stop action in simulation
        actions.remove(Directions.STOP)
        # try to avoid reverse as much as possible
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if reverse in actions and len(actions) > 1:
            actions.remove(reverse)
        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            value = self.evaluate(successor, Directions.STOP)
            values.append(value)
        return max(values) * decay

    def simulation(self, gameState, decay):
        # in the simulation, we only simulate for 2 depth for computing convenience
        visit_time = {}
        values = {}
        ucbs = {}
        successors = []
        actions = gameState.getLegalActions(self.index)
        # no stop action in simulation
        actions.remove('Stop')
        # try to avoid reverse as much as possible
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if reverse in actions and len(actions) > 1:
            actions.remove(reverse)
        visit_time[gameState] = 2
        values[gameState] = 0
        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            successors.append(successor)
            visit_time[successor] = 1
            value = self.evaluate(successor, Directions.STOP)
            values[successor] = value
            ucbs[successor] = self.getUCB(value, visit_time[gameState], visit_time[successor])
        # first expand all successors and get their values
        chosenState = max(ucbs, key=ucbs.get)
        visit_time[chosenState] += 1
        values[chosenState] += self.expand(chosenState, decay)

        visit_time[gameState] = 3
        for successor in successors:
            ucbs[successor] = self.getUCB(values[successor], visit_time[gameState], visit_time[successor])
        # then calculate the UCB of successors, try to get the most valuable one
        chosenState = max(ucbs, key=ucbs.get)
        while visit_time[chosenState] != 2:
            visit_time[chosenState] += 1
            values[chosenState] += self.expand(chosenState, decay)
            for successor in successors:
                ucbs[successor] = self.getUCB(values[successor], visit_time[gameState], visit_time[successor])
            chosenState = max(ucbs, key=ucbs.get)
        return values[chosenState] * decay

    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.agent.index)
        actions.remove(Directions.STOP)
        result = {}
        for action in actions:
            value = self.simulation(gameState.generateSuccessor(self.agent.index, action), 0.8)
            result[action] = value
        return max(result, key=result.get)


class OffensiveAction(MCTSActions):
    """
    Offensive actions of agents, use MCTS technique
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        visibleEnemies = []
        visiblePacman = []
        food_list = self.agent.getFood(successor).asList()
        capsule_list = self.agent.getCapsules(successor)

        for index in self.agent.getOpponents(gameState):
            enemy = successor.getAgentState(index)
            if enemy.getPosition() is not None:
                if not enemy.isPacman:
                    visibleEnemies.append(enemy)
                else:
                    visiblePacman.append(enemy)

        # feature1: distance to nearest food
        if len(food_list) > 0:
            min_dis_food = min([self.agent.getMazeDistance(pos, food) for food in food_list])
            features['distanceToFood'] = min_dis_food
        else:
            features['distanceToFood'] = 0

        # # feature2: distance to nearest capsule
        # if len(capsule_list) > 0:
        #     min_dis_capsule = min([self.agent.getMazeDistance(pos, capsule) for capsule in capsule_list])
        #     features['distanceToCapsule'] = min_dis_capsule
        # else:
        #     features['distanceToCapsule'] = 0

        # feature3: foods that have been eaten
        features['foodEaten'] = successor.getAgentState(self.index).numCarrying

        # feature4: current score
        features['score'] = self.agent.getScore(successor)

        # feature5: distance to closest ghost
        # if has visible enemies, get the closest one
        if len(visibleEnemies) > 0:
            # only the opponents that are within 5 are visible
            min_distance = 5 + 1
            min_dis_ghost = min([self.agent.getMazeDistance(pos, enemy.getPosition()) for enemy in visibleEnemies])
            if min_dis_ghost < min_distance:
                min_distance = min_dis_ghost
            features['distanceToGhost'] = min_distance
        # else ignore this feature
        else:
            features['distanceToGhost'] = 0

        # feature6: distance to closest pacman when we are at home
        # if has visible pacmans when we are at home, get the closest one
        if len(visiblePacman) > 0 and not successor.getAgentState(self.index).isPacman:
            min_distance = 5 + 1
            min_dis_pacman = min([self.agent.getMazeDistance(pos, pacman.getPosition()) for pacman in visiblePacman])
            if min_dis_pacman < min_distance:
                min_distance = min_dis_pacman
            features['distanceToPacman'] = min_distance
        else:
            features['distanceToPacman'] = 0

        # # feature7: distance to home
        # min_dis_home = min([self.agent.getMazeDistance(pos, boundary) for boundary in self.boundary])
        # # we only care whether can go back home when being pacman, in offensive action
        # if not successor.getAgentState(self.index).isPacman:
        #     features['distanceToHome'] = 0
        # else:
        #     features['distanceToHome'] = min_dis_home

        return features

    def getWeights(self, gameState, action):
        weights = util.Counter()
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        visiblePacman = []
        visibleEnemies = []

        foodEaten = successor.getAgentState(self.index).numCarrying

        for index in self.agent.getOpponents(successor):
            enemy = successor.getAgentState(index)
            if enemy.getPosition() is not None:
                if not enemy.isPacman:
                    visibleEnemies.append(enemy)
                else:
                    visiblePacman.append(enemy)

        # we always want to make the score as much as possible
        weights['score'] = 500

        # when we are at home and pacman has no capsule, try to kill it
        if not successor.getAgentState(self.index).isPacman:
            weights['distanceToPacman'] = -100
        else:
            weights['distanceToPacman'] = 0

        if len(visibleEnemies) == 0:
            # try to eat food and capsule, ignore the ghost
            weights['distanceToFood'] = -50
            # weights['distanceToCapsule'] = -25
            weights['distanceToGhost'] = 0
            weights['foodEaten'] = 5000
            # # more dangerous when going further, but should also consider taking more food
            # weights['distanceToHome'] = 25 - foodEaten * 15
        else:
            enemy = None
            min_dis = 10000
            for item in visibleEnemies:
                dis = self.agent.getMazeDistance(item.getPosition(), pos)
                if dis < min_dis:
                    min_dis = dis
                    enemy = item
            # when ghosts are scare of pacman, and we can see the enemy at this time
            if enemy.scaredTimer >= 5:
                # try to eat food as much as possible
                weights['distanceToFood'] = -50
                # now the capsule can be ignored
                # weights['distanceToCapsule'] = 0
                if self.agent.getMazeDistance(pos, enemy.getPosition()) <= 1:
                    weights['distanceToGhost'] = -5000
                else:
                    weights['distanceToGhost'] = -500
                weights['foodEaten'] = 1500
                # # try to get home, but should also consider taking more food
                # weights['distanceToHome'] = 25 - foodEaten * 15
            else:
                # try to eat food less
                weights['distanceToFood'] = -25
                # if has enemy chasing, try to eat capsule and kill the enemy
                # for enemy in visibleEnemies:
                # try to avoid the ghost as much as possible
                weights['distanceToGhost'] = 500
                weights['foodEaten'] = 50
                # # try to get home, should also consider taking more food, but less weight
                # weights['distanceToHome'] = -5 - foodEaten * 15

        return weights


class Tree:
    """
     a base class to compute actions that reach to goal state
    """

    def initial(self, pos):
        node = {}
        node['position'] = pos
        node['father'] = None
        node['action'] = None
        return node

    def expand(self, father_node, next_pos, action):
        node = {}
        node['position'] = next_pos
        node['father'] = father_node
        node['action'] = action
        return node

    def actionToGoal(self, goal):
        actions = []
        while goal['father'] is not None:
            actions.append(goal['action'])
            goal = goal['father']
        return list(reversed(actions))


##############################
# Astar back-home Action #
##############################
class BackHome:
    """
    a new class to implement back-home strategy, use astar heuristic
    """

    def __init__(self, agent, index, gameState):
        self.agent = agent
        self.index = index
        # used to store location of boundary
        self.boundary = []
        self.gameState = gameState

        if self.agent.red:
            mid_line = (gameState.data.layout.width - 2) // 2
        else:
            mid_line = (gameState.data.layout.width - 2) // 2 + 1

        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(mid_line, i):
                self.boundary.append((mid_line, i))

    def wrongSuccessor(self):
        wall = self.gameState.getWalls().asList()
        wrong = wall
        enemies = []
        enemy_area = []
        if self.agent.red:
            for i in range((self.gameState.data.layout.width - 2) // 2 + 1, self.gameState.data.layout.width - 1):
                for j in range(1, self.gameState.data.layout.height - 1):
                    enemy_area.append((i, j))
        else:
            for i in range(1, (self.gameState.data.layout.width - 2) // 2 + 1):
                for j in range(1, self.gameState.data.layout.height - 1):
                    enemy_area.append((i, j))

        for index in self.agent.getOpponents(self.agent.getCurrentObservation()):
            enemy = self.agent.getCurrentObservation().getAgentState(index)
            if enemy.getPosition() is not None:
                if not enemy.isPacman and not enemy.scaredTimer > 0:
                    enemies.append(enemy)

        if len(enemies) > 0:
            danger = [-1, 0, 1]
            for enemy in enemies:
                enemy_pos = enemy.getPosition()
                for i in danger:
                    if (enemy_pos[0] + i, enemy_pos[1]) in enemy_area:
                        wrong.append((enemy_pos[0] + i, enemy_pos[1]))
                    if (enemy_pos[0], enemy_pos[1] + i) in enemy_area:
                        wrong.append((enemy_pos[0], enemy_pos[1] + i))

        return wrong

    def getNextAction(self, pos):
        wrong = self.wrongSuccessor()
        successor = []
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        directions = ['North', 'South', 'East', 'West']
        direction = None
        for action in actions:
            next_pos = ((pos[0] + action[0]), (pos[1] + action[1]))
            if action == (0, 1):
                direction = directions[0]
            elif action == (0, -1):
                direction = directions[1]
            elif action == (1, 0):
                direction = directions[2]
            elif action == (-1, 0):
                direction = directions[3]
            if next_pos not in wrong:
                successor.append((next_pos, direction))
        return successor

    def astar(self, goal):
        search = Tree()
        root_pos = self.agent.getCurrentObservation().getAgentState(self.index).getPosition()
        open = util.PriorityQueue()
        open.push((search.initial(root_pos), 0), 0)
        closed = []
        best_g = {}
        while not open.isEmpty():
            cur_node, cur_g = open.pop()
            if cur_node['position'] not in closed or cur_g < best_g[cur_node['position']]:
                closed.append(cur_node['position'])
                best_g[cur_node['position']] = cur_g
                if cur_node['position'] == goal:
                    if cur_node["action"] is None:
                        continue
                    return search.actionToGoal(cur_node)[0]
                for successor in self.getNextAction(cur_node['position']):
                    next_node = search.expand(cur_node, successor[0], successor[1])
                    if next_node['position'] not in closed:
                        actions = search.actionToGoal(next_node)
                        # 用曼哈顿做heuristic
                        dis = abs(goal[0] - next_node['position'][0]) + abs(goal[1] - next_node['position'][1])
                        priority = len(actions) + dis
                        open.push((next_node, len(actions)), priority)
                        best_g[next_node['position']] = len(actions)
        return 'Stop'

    def getTarget(self):
        pos = self.agent.getCurrentObservation().getAgentPosition(self.index)
        min_dis = 10000
        target = None
        enemies = []
        for index in self.agent.getOpponents(self.agent.getCurrentObservation()):
            enemy = self.agent.getCurrentObservation().getAgentState(index)
            if enemy.getPosition() is not None:
                if not enemy.isPacman and not enemy.scaredTimer > 0:
                    enemies.append(enemy)
        for boundary in self.boundary:
            dis = self.agent.getMazeDistance(boundary, pos)
            if dis < min_dis:
                min_dis = dis
                target = boundary
        capsule_list = self.agent.getCapsules(self.agent.getCurrentObservation())
        if len(capsule_list) == 0 or self.agent.getCurrentObservation().getAgentState(self.index).numCarrying >= 7:
            return target
        else:
            if len(enemies) == 0:
                return target
            else:
                for capsule in capsule_list:
                    if self.agent.getMazeDistance(capsule, pos) <= 2:
                        target = capsule
                return target

    def chooseAction(self):
        return self.astar(self.getTarget())


##########
# Agents #
##########

class OffensiveAgent(CaptureAgent):

    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        enemies = []
        for index in self.getOpponents(self.getCurrentObservation()):
            enemy = self.getCurrentObservation().getAgentState(index)
            if enemy.getPosition() is not None:
                if not enemy.isPacman and not enemy.scaredTimer > 0:
                    enemies.append(enemy)

        min_dis = 1000
        for enemy in enemies:
            dis = self.getMazeDistance(enemy.getPosition(),
                                       self.getCurrentObservation().getAgentState(self.index).getPosition())
            if dis < min_dis:
                min_dis = dis

        if gameState.getAgentState(self.index).numCarrying >= 7 or len(self.getFood(gameState).asList()) <= 2:
            return BackHome(self, self.index, gameState).chooseAction()
        elif min_dis <= 4 and gameState.getAgentState(self.index).isPacman:
            return BackHome(self, self.index, gameState).chooseAction()
        else:
            return OffensiveAction(self, self.index, gameState).chooseAction(gameState)


class DefensiveAgent(CaptureAgent):
    """
    defensive agent, use simple heuristic search
    """
    # register the initial state
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.boundary = []

        if self.red:
            mid_line = (gameState.data.layout.width - 2) // 2
        else:
            mid_line = (gameState.data.layout.width - 2) // 2 + 1

        for i in range(5, gameState.data.layout.height - 5):
            if not gameState.hasWall(mid_line, i):
                self.boundary.append((mid_line, i))

        self.defendFoods = self.getDefendFoods(gameState)
        self.target = None
        self.previousFoodList = self.getFoodYouAreDefending(gameState).asList()

    def getDefendFoods(self, gameState):
        food_list = self.getFoodYouAreDefending(gameState).asList()
        defend = []

        for boundary in self.boundary:
            min_dis = 10000
            target = None
            for food in food_list:
                if self.getMazeDistance(food, boundary) < min_dis:
                    min_dis = self.getMazeDistance(food, boundary)
                    target = food
            defend.append(target)

        return defend

    def getTarget(self, gameState):
        pos = gameState.getAgentState(self.index).getPosition()
        cur_food = self.getFoodYouAreDefending(gameState).asList()

        if pos == self.target:
            goal = None
        else:
            goal = self.target

        enemies = []
        for index in self.getOpponents(gameState):
            enemy = gameState.getAgentState(index)
            if enemy.getPosition() is not None:
                if enemy.isPacman:
                    enemies.append(enemy)

        # defender should always be ghost
        if not gameState.getAgentState(self.index).isPacman:
            if len(enemies) > 0:
                min_dis = 10000
                target_pos = None
                for enemy in enemies:
                    if self.getMazeDistance(pos, enemy.getPosition()) < min_dis:
                        min_dis = self.getMazeDistance(pos, enemy.getPosition())
                        target_pos = enemy.getPosition()
                if gameState.getAgentState(self.index).scaredTimer > 0 and self.getMazeDistance(pos, target_pos) == 2:
                    actions = gameState.getLegalActions(self.index)
                    actions.remove('Stop')
                    valid_move = []
                    for action in actions:
                        next_pos = gameState.generateSuccessor(self.index, action).getAgentState(
                            self.index).getPosition()
                        if not self.getMazeDistance(target_pos, next_pos) <= 1:
                            valid_move.append(next_pos)
                    if not len(valid_move) == 0:
                        goal = random.choice(valid_move)
                else:
                    goal = target_pos
            else:
                if len(list(set(self.previousFoodList))) > len(list(set(cur_food))):
                    foodEaten = []
                    for food in set(self.previousFoodList):
                        if food not in set(cur_food):
                            foodEaten.append(food)
                    goal = random.choice(foodEaten)
                    self.previousFoodList = cur_food

        if goal is None:
            if 4 >= len(cur_food) > 0:
                goal = random.choice(cur_food)
            else:
                self.defendFoods = self.getDefendFoods(gameState)
                defendArea = self.defendFoods
                defendArea.extend(self.boundary)
                goal = random.choice(defendArea)

        return goal

    def search(self, goal, gameState):
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        min_dis = 10000
        chosen_action = random.choice(actions)

        for action in actions:
            successor = gameState.generateSuccessor(self.index, action)
            next_pos = successor.getAgentState(self.index).getPosition()
            if not successor.getAgentState(self.index).isPacman:
                distance = self.getMazeDistance(next_pos, self.target)
            else:
                enemies = []
                for index in self.getOpponents(gameState):
                    enemy = gameState.getAgentState(index)
                    if enemy.getPosition() is not None:
                        if not enemy.isPacman and not enemy.scaredTimer > 0:
                            enemies.append(enemy)
                if len(enemies) > 0:
                    min_distance = 10000
                    for enemy in enemies:
                        if self.getMazeDistance(enemy.getPosition(), next_pos) < min_distance:
                            min_distance = self.getMazeDistance(enemy.getPosition(), next_pos)
                    if min_distance <= 2:
                        continue
                    else:
                        distance = self.getMazeDistance(next_pos, self.target)
                else:
                    distance = self.getMazeDistance(next_pos, self.target)

            if distance < min_dis:
                min_dis = distance
                chosen_action = action

        if chosen_action is None:
            return 'Stop'
        return chosen_action

    def chooseAction(self, gameState):
        self.target = self.getTarget(gameState)
        return self.search(self.target, gameState)
