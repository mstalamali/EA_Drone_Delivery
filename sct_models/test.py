from sct import SCT

class Robot:
    def __init__(self, path):
        self.sct = SCT(path)
        self.isCapableTask = False
        self.isWonBid = False
        self.isCondReturn = False

        self.isBidding = False

    # Register callback functions to the generator player
    def add_callbacks(self):

        # Automatic addition of callbacks
        # 1. Get list of events and list specifying whether an event is controllable or not.
        # 2. For each event, check controllable or not and add callback.

        events, controllability_list = self.sct.get_events()

        for event, index in events.items():
            is_controllable = controllability_list[index]
            stripped_name = event.split('EV_', 1)[1]    # Strip preceding string 'EV_'

            if is_controllable: # Add controllable event
                func_name = '_callback_{0}'.format(stripped_name)
                func = getattr(self, func_name)
                self.sct.add_callback(self.sct.EV[event], func, None, None)
            else:   # Add uncontrollable event
                func_name = '_check_{0}'.format(stripped_name)
                func = getattr(self, func_name)
                self.sct.add_callback(self.sct.EV[event], None, func, None)

    # Uncontrolled event callbacks
    def _check_capableTask(self, data):
        if(self.isCapableTask and not self.isBidding):
            print('Triggered capableTask')
            return True
        return False
    
    def _check_notCapableTask(self, data):
        if(self.isCapableTask and not self.isBidding):
            return False
        print('Triggered notCapableTask')
        return True
    
    def _check_wonBid(self, data):
        if(self.isWonBid and self.isBidding):
            print('Triggered wonBid')
            self.isBidding = False
            return True
        return False    
    
    def _check_lostBid(self, data):
        if(not self.isWonBid and self.isBidding):
            print('Triggered lostBid')
            self.isBidding = False
            return True
        return False
    
    def _check_condReturn(self, data):
        if(self.isCondReturn):
            print('Triggered condReturn')
        return self.isCondReturn
    
    def _check_notCondReturn(self, data):
        return not self.isCondReturn

    # Controllable event callbacks
    def _callback_bid(self, data):
        print("Made a bid")
        self.isBidding = True

    def _callback_deliver(self, data):
        print("Delivering")
        self.isCapableTask = False
        self.isWonBid = False
        self.isCondReturn = False

        self.isBidding = False

    def _callback_return(self, data):
        print("Returning")


# main func
if __name__ == "__main__":
    sct_path = 'controller.yaml'
    drone1 = Robot(sct_path)

    drone1.add_callbacks()

    print('------')
    drone1.sct.run_step()

    print('------')
    drone1.isCapableTask = True
    drone1.sct.run_step()

    print('------')
    drone1.isWonBid = True
    drone1.sct.run_step()

    print('------')
    drone1.isCondReturn = True
    drone1.sct.run_step()

    print('------')
    drone1.isCondReturn = False
    drone1.isCapableTask = True
    drone1.sct.run_step()

    print('------')
    drone1.isWonBid = False
    drone1.isCapableTask = False
    drone1.sct.run_step()

    print('------')
    drone1.isCapableTask = True
    drone1.sct.run_step()