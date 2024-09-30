from helpers.utils import CommunicationState

# class defining the robot communication (it allows the robot to gather others' bids)
class CommunicationSession:
    def __init__(self, client, neighbors):
        self._client = client
        self._bids = [n._bid for n in neighbors if n.comm_state == CommunicationState.OPEN]
        self._bidders = [n.id for n in neighbors if n.comm_state == CommunicationState.OPEN]

    def get_bids(self):
        return [self._bids,self._bidders]