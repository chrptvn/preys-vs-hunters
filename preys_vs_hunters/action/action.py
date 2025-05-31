from enum import Enum

from preys_vs_hunters.goal.Goal import GoalType


class ActionType(Enum):
    WAIT = 0
    SET_GOAL = 1
    MOVE = 2



class SetGoal:
    def __init__(self, goal_type: GoalType, entity_id):
        pass