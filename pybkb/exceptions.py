"""
The exception classes for PyBKB
"""

class InternalBKBError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class INodeInBKBError(Exception):
    def __init__(
            self,
            component_name,
            state_name,
            message="I-node exists in BKB."
            ):
        """Exception raised when user adds an I-node that already exists in the BKB.

        Args:
            :param component_name: Name of the I-node component that is already in the BKB.
            :type component_name: str
            :param state_name: Name of the I-node state that is already in the BKB.
            :type state_name: str
        
        Kwargs:
            :param message: The error message that will be returned to user.
            :type message: str
        """
        self.component_name = component_name
        self.state_name = state_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{(self.component_name, self.state_name)}: {self.message}'

class NoINodeError(Exception):
    def __init__(
            self,
            component_name,
            state_name,
            message="I-node does not exist in BKB."
            ):
        """Exception raised when user adds an I-node that already exists in the BKB.

        Args:
            :param component_name: Name of the I-node component that is already in the BKB.
            :type component_name: str
            :param state_name: Name of the I-node state that is already in the BKB.
            :type state_name: str
        
        Kwargs:
            :param message: The error message that will be returned to user.
            :type message: str
        """
        self.component_name = component_name
        self.state_name = state_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{(self.component_name, self.state_name)}: {self.message}'

class SNodeProbError(Exception):
    def __init__(
            self,
            prob,
            message="S-node probability not between 0 and 1."
            ):
        """Exception raised when user adds illegal S-node.

        Args:
            :param prob: The probability for the passed S-node.
            :type component_name: float
        """
        self.prob = prob
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'Invalid S-node probability {self.prob}: {self.message}'


class BKBNotMutexError(Exception):
    def __init__(
            self,
            snode_idx1,
            snode_idx2,
            message="BKB is not mutex."
            ):
        """Exception raised when a BKB is not Mutex.

        Args:
            :param snode_idx1: Index of S-node that is not mutex with the other S-node.
            :type snode_idx1: int
            :param snode_idx2: Index of S-node that is not mutex with the other S-node.
            :type snode_idx2: int
        
        Kwargs:
            :param message: The error message that will be returned to user.
            :type message: str
        """
        self.snode_idx1 = snode_idx1
        self.snode_idx2 = snode_idx2
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'S-node with index {self.snode_idx1} is not mutex with S-node with index {self.snode_idx2}: {self.message}'
