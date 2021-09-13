from captureAgents import CaptureAgent
import random, time, util
import copy
from game import Directions
from util import nearestPoint
from collections import defaultdict
import game
import copy


# python capture.py -b astar_new -r MCTS_new -l RANDOM9968    pacman loops
# python capture.py -r baselineTeam -b astar_new -l RANDOM3971   debug time and illegal
# python capture.py -r baselineTeam -b astar_new -l RANDOM38   check normal back home


# python capture.py -b astar_new -r MCTS_new -l RANDOM996
# debug attackloop

# python capture.py -r astar_new -b AAA -l RANDOM5267
# avoid powerful pacman


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

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
            if target is not None:
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


class OffensiveAgent(CaptureAgent):

    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''
        self.boundary = []
        if self.red:
            mid_line = (gameState.data.layout.width - 2) // 2
            enemy_line = mid_line + 1
        else:
            mid_line = (gameState.data.layout.width - 2) // 2 + 1
            enemy_line = mid_line - 1
        # consider opponent's wall at the same time
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(mid_line, i) and not gameState.hasWall(enemy_line, i):
                self.boundary.append((mid_line, i))

        # maximum 300
        self.step = 0
        # avoid loop
        self.iterate = 0
        # safe distance with enemy
        self.safe_dis = 5 + 3
        # avoid loop, record the steps of each pos
        self.step_pos = defaultdict(int)
        self.goal = None

    # same function as in baselineTeam
    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def initialAction(self, gameState):
        pos = gameState.getAgentState(self.index).getPosition()
        capsule_list = self.getCapsules(gameState)
        self.step += 1
        # store all enemies, no matter ghost or pacman
        enemies = []
        for index in self.getOpponents(gameState):
            enemy = gameState.getAgentState(index)
            if enemy.getPosition() is not None:
                enemies.append(enemy)
        return pos, capsule_list, enemies

    def enterBoundary(self, gameState, actions):
        pos, capsule_list, enemies = self.initialAction(gameState)
        self.cur_foods = self.getFood(gameState).asList()
        if self.goal is None or not self.goal in self.boundary:
            self.goal = random.choice(self.boundary)
        if pos in self.boundary and self.step - self.step_pos[pos] <= 10:
            self.goal = random.choice(self.boundary)
        if pos in self.boundary and self.step - self.step_pos[pos] > 10:
            self.step_pos[pos] = self.step
            min_dis = 10000
            for enemy in enemies:
                if self.getMazeDistance(pos, enemy.getPosition()) < min_dis:
                    min_dis = self.getMazeDistance(pos, enemy.getPosition())
            # if not dangerous, enter directly
            if min_dis >= 5:
                if self.red:
                    return 'East'
                return 'West'
            else:
                # see if distance too close
                if self.red:
                    successor = self.getSuccessor(gameState, 'East')
                else:
                    successor = self.getSuccessor(gameState, 'West')
                next_pos = successor.getAgentState(self.index).getPosition()
                for enemy in enemies:
                    if self.getMazeDistance(next_pos, enemy.getPosition()) <= 2 or \
                            self.getMazeDistance(next_pos, enemy.getPosition()) > 10:
                        boundary = copy.deepcopy(self.boundary)
                        boundary.remove(pos)
                        # if has no choice now, back 1 step
                        if len(boundary) == 0:
                            for a in actions:
                                successor = self.getSuccessor(gameState, a)
                                next_pos = successor.getAgentState(self.index).getPosition()
                                if self.getMazeDistance(next_pos, enemy.getPosition()) > 0:
                                    return a
                            return 'Stop'
                        self.goal = random.choice(boundary)
                        return self.search(gameState, self.goal, actions)
                if self.red:
                    return 'East'
                return 'West'
        else:
            # if has powerful pacman use avoid search
            if gameState.getAgentState(self.index).scaredTimer > 0:
                return self.avoid_search(gameState, self.goal, actions)
            else:
                action = self.search(gameState, self.goal, actions)
                return action

    # helper function for avoid_search
    def getClosestInvader(self, gameState, pos):
        invaders = []
        closest_invader = None
        for index in self.getOpponents(gameState):
            enemy = gameState.getAgentState(index)
            if enemy.getPosition() is not None and enemy.isPacman:
                invaders.append(enemy)
        dis_toInvader = 10000
        for invader in invaders:
            if self.getMazeDistance(pos, invader.getPosition()) < dis_toInvader:
                dis_toInvader = self.getMazeDistance(pos, invader.getPosition())
                closest_invader = invader.getPosition()
        return closest_invader

    def avoid_search(self, gameState, goal, actions, attack=True):
        pos = gameState.getAgentState(self.index).getPosition()
        avoid_pos = self.getClosestInvader(gameState, pos)
        min_dis = 10000
        chosen_action = None
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            next_pos = successor.getAgentState(self.index).getPosition()
            if attack and successor.getAgentState(self.index).isPacman:
                continue
            if avoid_pos is not None and self.getMazeDistance(next_pos, avoid_pos) <= 1:
                continue
            if self.getMazeDistance(next_pos, goal) < min_dis:
                min_dis = self.getMazeDistance(next_pos, goal)
                chosen_action = action
        if chosen_action is None:
            return 'Stop'
        else:
            return chosen_action

    def attackMode(self, gameState, actions):
        pos, capsule_list, enemies = self.initialAction(gameState)
        # how many foods was eaten since last enter
        self.foodEaten = len(self.cur_foods) - len(self.getFood(gameState).asList())

        # find closest enemy, only ghost enemy
        closest_enemy = None
        ghosts = []
        for enemy in enemies:
            if not enemy.isPacman:
                ghosts.append(enemy)
        dis_toEnemy = 10000
        for ghost in ghosts:
            if self.getMazeDistance(pos, ghost.getPosition()) < dis_toEnemy:
                dis_toEnemy = self.getMazeDistance(pos, ghost.getPosition())
                closest_enemy = ghost

        # find closest home
        closest_home = None
        dis_toHome = 10000
        for boundary in self.boundary:
            if self.getMazeDistance(pos, boundary) < dis_toHome:
                dis_toHome = self.getMazeDistance(pos, boundary)
                closest_home = boundary

        # if has only 2 foods left or the dis to closest home equals to the left time, go back home
        # use BFS, not closest home, can easily stop
        if len(self.getFood(gameState).asList()) == 2 or dis_toHome >= 299 - self.step:
            return self.BFS(gameState, self.boundary, pos)

        # if not dangerous, eat foods
        if len(ghosts) == 0 or dis_toEnemy > self.safe_dis:
            score = self.getScore(gameState)
            if score >= 1:
                action, des, cost = self.astar(gameState, 1)
            else:
                action, des, cost = self.astar(gameState, 2)
            return action

        scared = False
        if closest_enemy.scaredTimer > 0:
            scared = True
        # if enemy scared, eat foods
        if scared:
            action, des, cost = self.astar(gameState, 2)
            live = self.boundary + capsule_list
            dis_toHome = 10000
            for boundary in self.boundary:
                if self.getMazeDistance(des, boundary) < dis_toHome:
                    dis_toHome = self.getMazeDistance(des, boundary)
                    closest_home = boundary

            closest_life = None
            dis_toLife = 10000
            for item in live:
                if self.getMazeDistance(des, item) < dis_toLife:
                    dis_toLife = self.getMazeDistance(des, item)
                    closest_life = item

            # if cost to food plus time back home equals to left time, go back home
            if cost + dis_toHome >= 300 - self.step:
                self.goal = closest_home
                return self.search(gameState, self.goal, actions, False)
            # if cost to food plus dis to safe place, go to food
            if cost + dis_toLife < closest_enemy.scaredTimer:
                return action
            else:
                self.goal = closest_life
                return self.search(gameState, self.goal, actions, False)
        else:
            # when not scared, use BFS
            # if has eaten less than 2 foods, go to capsule
            live = self.boundary + capsule_list
            if self.foodEaten >= 2:
                goals = live
            else:
                if len(capsule_list) > 0:
                    goals = capsule_list
                    action = self.BFS(gameState, goals, pos)
                    if action == 'Stop':
                        if self.safe_dis - 1 / 4 <= 2:
                            self.safe_dis = 2
                        else:
                            self.safe_dis -= 1 / 4
                        goals = self.boundary
                else:
                    if self.safe_dis - 1 / 4 <= 2:
                        self.safe_dis = 2
                    else:
                        self.safe_dis -= 1 / 4
                    goals = self.boundary
            action = self.BFS(gameState, goals, pos)
            action = self.attackLoop(gameState, action, pos)
            return action

    # pacman can easily loop when meet not scared pacman
    # if has no wall, go back for 1 step
    def attackLoop(self, gameState, action, pos):
        # remain stop here
        if self.getPreviousObservation() is not None:
            previous_direction = self.getPreviousObservation().getAgentState(self.index).configuration.direction
        else:
            previous_direction = None

        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if reverse == previous_direction:
            self.iterate += 1
        else:
            self.iterate = 0

        # see if ghost is at east or west
        enemy_pos = self.findClosestEnemy(gameState, pos)
        # if at east, back west
        if enemy_pos[0] - pos[0] > 0:
            back = 'West'
        elif enemy_pos[0] - pos[0] < 0:
            back = 'East'
        else:
            back = action

        if self.iterate >= 5:
            # when back and has no wall
            if back in gameState.getLegalActions(self.index):
                action = back
            else:
                action = random.choice(gameState.getLegalActions(self.index))

        return action

    # avoid loop within 5 steps
    def removeLoop(self, gameState, actions):
        actions.remove('Stop')
        if self.getPreviousObservation() is not None:
            previous_direction = self.getPreviousObservation().getAgentState(self.index).configuration.direction
        else:
            previous_direction = None

        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if reverse == previous_direction:
            self.iterate += 1
        else:
            self.iterate = 0

        if self.iterate >= 5:
            actions.remove(reverse)

        return actions

    def search(self, gameState, goal, actions, attack=True):
        min_dis = 10000
        chosen_action = None
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            next_pos = successor.getAgentState(self.index).getPosition()
            if attack and successor.getAgentState(self.index).isPacman:
                continue
            if self.getMazeDistance(next_pos, goal) < min_dis:
                min_dis = self.getMazeDistance(next_pos, goal)
                chosen_action = action
        if chosen_action is None:
            return 'Stop'
        else:
            return chosen_action

    # use heuristic search to eat foods，when safe, return an action，destination and cost
    def astar(self, gameState, goals):
        open_list = util.PriorityQueue()
        closed = []
        best_g = {}
        actions = []
        cur_goals = goals
        initial_foods = self.getFood(gameState).asList()
        if len(initial_foods) < 2:
            cur_goals = 1
        # if has eaten food, eat one
        if self.foodEaten >= 1:
            cur_goals = 1
        pos = gameState.getAgentState(self.index).getPosition()
        best_g[pos] = 0
        open_list.push((gameState, actions, 0), 0 + self.h(gameState, cur_goals, initial_foods))

        while not open_list.isEmpty():
            cur_state, actions, cost = open_list.pop()
            cur_pos = cur_state.getAgentState(self.index).getPosition()
            if cur_pos not in closed or cost < best_g[cur_pos]:
                closed.append(cur_pos)
                best_g[cur_pos] = cost
                # see how many foods has eaten in this astar, if reached goals, return
                cur_foods = self.getFood(cur_state).asList()
                if len(initial_foods) - len(cur_foods) >= cur_goals:
                    if len(actions) == 0:
                        return 'Stop', cur_pos, cost
                    return actions[0], cur_pos, cost
                for a in cur_state.getLegalActions(self.index):
                    successor = self.getSuccessor(cur_state, a)
                    if successor.getAgentState(self.index).isPacman:
                        if successor.getAgentState(self.index).getPosition() not in closed:
                            eaten = len(initial_foods) - len(cur_foods)
                            open_list.push((successor, actions + [a], cost + 1), cost + 1 + self.h(
                                successor, cur_goals - eaten, cur_foods))

        if goals > 0:
            goals -= 1
            return self.astar(gameState, goals)
        else:
            a = random.choice(gameState.getLegalActions(self.index))
            if a is None:
                return 'Stop', pos, 0
            return a, pos, 0

    # heuristic function, find closest food
    def h(self, gameState, goals, foods):
        pos = gameState.getAgentState(self.index).getPosition()
        # if eat one, choose the closest food, if two, use combined dis
        if goals == 1:
            min_dis = 10000
            for food in foods:
                if self.getMazeDistance(pos, food) < min_dis:
                    min_dis = self.getMazeDistance(pos, food)
            return min_dis
        else:
            min_dis1 = min_dis2 = 10000
            for food in foods:
                if self.getMazeDistance(pos, food) < min_dis1:
                    min_dis2 = min_dis1
                    min_dis1 = self.getMazeDistance(pos, food)
                elif self.getMazeDistance(pos, food) < min_dis2:
                    min_dis2 = self.getMazeDistance(pos, food)
            return min_dis2 + min_dis1

    # helper function for BFS, only avoid enemies that are not scared
    def findClosestEnemy(self, gameState, pos):
        enemies = []
        target = None
        for index in self.getOpponents(gameState):
            enemy = gameState.getAgentState(index)
            if enemy.getPosition() is not None and not enemy.scaredTimer > 0:
                if not enemy.isPacman:
                    enemies.append(enemy)
        if len(enemies) > 0:
            min_dis = 1000
            for enemy in enemies:
                dis = self.getMazeDistance(enemy.getPosition(), pos)
                if dis < min_dis:
                    min_dis = dis
                    target = enemy.getPosition()
        return target

    def BFS(self, gameState, goals, pos):
        avoid_pos = self.findClosestEnemy(gameState, pos)
        open = util.Queue()
        closed = []
        actions = []
        open.push((gameState, actions))
        while not open.isEmpty():
            cur_state, actions = open.pop()
            cur_pos = cur_state.getAgentState(self.index).getPosition()
            if cur_pos not in closed:
                closed.append(cur_pos)
                if cur_pos in goals:
                    if len(actions) == 0:
                        return 'Stop'
                    else:
                        return actions[0]
                for a in cur_state.getLegalActions(self.index):
                    successor = self.getSuccessor(cur_state, a)
                    next_pos = successor.getAgentState(self.index).getPosition()
                    if next_pos not in closed:
                        if avoid_pos is not None and self.getMazeDistance(avoid_pos, next_pos) <= 1:
                            continue
                        open.push((successor, actions + [a]))

        return 'Stop'

    def chooseAction(self, gameState):
        """
        Picks among actions for defending.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        new_actions = self.removeLoop(gameState, actions)

        if not gameState.getAgentState(self.index).isPacman:
            action = self.enterBoundary(gameState, new_actions)
            return action
        else:
            action = self.attackMode(gameState, new_actions)
            return action
