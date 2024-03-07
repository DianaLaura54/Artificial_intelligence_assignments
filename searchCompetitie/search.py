# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import random
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def randomSearch(problem) :
    current = problem.getStartState()
    solution = []
    while (not (problem.isGoalState(current))) :
        succ = problem.getSuccessors(current)
        no_of_successor = len ( succ )
        random_succ_index = int(random.random() * no_of_successor)
        next = succ[random_succ_index]
        current = next[0]
        solution.append(next[1])
    print "The solution is ", solution
    return solution

def depthFirstSearch(problem):
    """
    Search the deepest pozities in the search tree first.
    Your search algorithm needs to return a list of directie that reaches the
    goal. Make sure to implement a graph search algorithm.
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
 """
    print "Start dfs:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    solution = []

    startingLocation = problem.getStartState()
    if(problem.isGoalState(startingLocation)):
        print "The solution is", solution
        return solution

    stiva = util.Stack()
    visited = []

    stiva.push((startingLocation, []))
    while not stiva.isEmpty():
        current, solution = stiva.pop()

        if ( not ( current in visited ) ):
            visited.append(current)

            if( problem.isGoalState(current)):
                print "The solution is", solution 
                return solution


            successors = problem.getSuccessors(current) 

            for nextLocation, nextDirection, cost in successors:
                stiva.push((nextLocation, solution + [nextDirection]))


    print "The solution is", solution
    return solution
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest pozities in the search tree first."""
    "*** YOUR CODE HERE ***"
    print "Start bfs:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    solution = []

    startingLocation = problem.getStartState()
    if(problem.isGoalState(startingLocation)):
        print "The solution is", solution
        return solution

    coada = util.Queue()
    visited = []

    coada.push((startingLocation, []))
    while not coada.isEmpty():
        current, solution = coada.pop()

        if not (current in visited) :

            visited.append(current)

            if problem.isGoalState(current) :
                print "The solution is", solution
                return solution

            succ = problem.getSuccessors(current)

            for nextLocation, nextDirection, cost in succ:
                coada.push((nextLocation, solution + [nextDirection]))

    print "The solution is", solution
    return solution
    util.raiseNotDefined()


def aStarSearch(problem, weight, heuristic):
    """Search the pozitie that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    print "Start aStar:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    startingLocation = problem.getStartState()
    solution = []

    if problem.isGoalState(startingLocation):
        print "The solution is ", solution
        return solution
    
    visited = []

    pCoada = util.PriorityQueue()
    pCoada.push((startingLocation, [],0), 0)

    while not pCoada.isEmpty():
        current, solution, currentCost = pCoada.pop()

        if not (current in visited):
            visited.append(current)

            if problem.isGoalState(current):
                print "The solution is", solution
                return solution

            succ = problem.getSuccessors(current)
            for nextLocation, nextDirection, cost in succ:
                newCost = currentCost + cost
                heuristicCost = newCost + heuristic(nextLocation, problem)
                pCoada.push((nextLocation, solution + [nextDirection], newCost), heuristicCost)


    util.raiseNotDefined()
    
def aStarSearchw(problem, weight, heuristic):
    """Search the pozitie that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    print "Start aStar:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    startingLocation = problem.getStartState()
    solution = []

    if problem.isGoalState(startingLocation):
        print "The solution is ", solution
        return solution

    visited = []

    pCoada = util.PriorityQueue()
    pCoada.push((startingLocation, [],0), 0)

    
    while not pCoada.isEmpty():
        current, solution, currentCost = pCoada.pop()

        if not (current in visited):
            visited.append(current)

            if problem.isGoalState(current):
                print "The solution is", solution
                return solution

            succ = problem.getSuccessors(current)
            for nextLocation, nextDirection, cost in succ:
                newCost = currentCost + cost
                heuristicCost = newCost + float(weight)*heuristic(nextLocation, problem)
                pCoada.push((nextLocation, solution + [nextDirection], newCost), heuristicCost)


    util.raiseNotDefined()

def Fringe(problem,weight, heuristic):
    print("Start A* search:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    startingLocation = problem.getStartState()
    fringe = util.PriorityQueue()
    visited = []
    paths = []

    fringe.push((startingLocation, paths), heuristic(startingLocation, problem))

    while not fringe.isEmpty():
        state, solution= fringe.pop()

        if problem.isGoalState(state):
            return solution 

        if state not in visited:
            visited.append(state)
            successors = problem.getSuccessors(state)

            for successor in successors:
                next_state, action, step_cost = successor
                new_path = solution + [action]
                total_cost = len(new_path)+ heuristic(next_state, problem)  
                fringe.push((next_state, new_path), total_cost)

    util.raiseNotDefined()
    
def uniformCostSearch(problem):
    solution = []
    startingLocation = problem.getStartState()

    if problem.isGoalState(startingLocation):
        print "The solution is", solution
        return solution

    coadaPrioritara = util.PriorityQueue()
    coadaPrioritara.push((startingLocation, [], 0), 0)

    visited = []

    while not coadaPrioritara.isEmpty():

        current, solution, currentCost = coadaPrioritara.pop()
        if not (current in visited):
            visited.append(current)
            
            if problem.isGoalState(current):
                print "The solution is", solution
                return solution


            succ = problem.getSuccessors(current)

            for nextLocation, nextDirection, cost in succ:
                priority = currentCost + cost
                coadaPrioritara.push((nextLocation, solution + [nextDirection], priority), priority)        

    print "The solution is", solution
    return solution

    util.raiseNotDefined()

def iterativeDeepeningSearch(problem):
    print "Start IDS:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    startingLocation = problem.getStartState()
    solution = []

    stiva = util.Stack()
    adancime = 1

    while True:
        visited = []

        stiva.push((startingLocation,[],0))
        (current, solution , cost) = stiva.pop()
        visited.append(current)
    
        while not problem.isGoalState(current): 
            successors = problem.getSuccessors(current)
            for nextLocation, nextDirection, nextCost in successors:
                if (not nextLocation in visited) and (cost + nextCost <= adancime): 
                    stiva.push((nextLocation, solution + [nextDirection], cost + nextCost)) 
                    visited.append(nextLocation)

            if stiva.isEmpty():
                break

            (current, solution, cost) = stiva.pop()

        if problem.isGoalState(current):
            print "The solution is", solution
            return solution

        adancime += 1 

    print "The solution is", solution
    return solution
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
rs = randomSearch
ids = iterativeDeepeningSearch
astarw= aStarSearchw
fringe=Fringe
