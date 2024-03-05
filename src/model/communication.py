import copy

from model.navigation import Location, Target
from helpers.utils import rotate, CommunicationState

import numpy as np

class CommunicationSession:
    def __init__(self, client, neighbors):
        self._client = client
        self._bids = [n._bid for n in neighbors if n.comm_state == CommunicationState.OPEN]
        self._bidders = [n.id for n in neighbors if n.comm_state == CommunicationState.OPEN]

    def get_best_bid(self):
        if len(self._bids)>0:
            index=np.argmax(self._bids)
            max_value = self._bids[index]

            if self._bids.count(self._bids[index]) == 1:
                return [max_value, self._bidders[index]]
            else:

                bidders_with_max_bid = [self._bidders[i] for i in range(len(self._bidders)) if self._bids[i] == max_value]
                return [max_value, max(bidders_with_max_bid)]
        else:
            return [-1,-1]


    # def get_metadata(self, location):
    #     metadata = {n_id: {
    #         "age": n.get_info_from_behavior(location).get_age(),
    #     } for n_id, n in self._neighbors.items() if
    #         n.get_info_from_behavior(location) is not None and n.get_info_from_behavior(location).is_valid()}
    #     return metadata

    # def make_transaction(self, neighbor_id, location) -> Target:
    #     target = copy.deepcopy(self._neighbors[neighbor_id].get_info_from_behavior(location))
    #     if target is None:
    #         raise NoInformationSoldException
    #     target.rotate(self._neighbors[neighbor_id].orientation - self._client.orientation)
    #     transaction = Transaction(self._client.id,
    #                               neighbor_id,
    #                               location,
    #                               get_orientation_from_vector(target.get_distance()),
    #                               None)
    #     self._client.record_transaction(transaction)
    #     self._client.communication_happened()
    #     self._neighbors[neighbor_id].communication_happened()
    #     return target

    # def get_distance_from(self, neighbor_id):
    #     distance = self._neighbors[neighbor_id].pos - self._client.pos
    #     return rotate(distance, -self._client.orientation)

    # def get_own_reward(self):
    #     return self._client.reward()
