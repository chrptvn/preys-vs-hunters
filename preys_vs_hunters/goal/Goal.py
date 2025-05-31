from enum import Enum


class GoalType(Enum):
    STAY = 0
    CHASE = 1
    ESCAPE = 2


class Goal:
    def __init__(self, type: GoalType, target_id: int = None):
        self.type = type
        self.target_id = target_id


class Stay(Goal):
    def __init__(self):
        super().__init__(GoalType.CHASE, -1)


class Chase(Goal):
    def __init__(self, target_id: int):
        super().__init__(GoalType.CHASE, target_id)


class Escape(Goal):
    def __init__(self, target_id: int):
        super().__init__(GoalType.ESCAPE, target_id)