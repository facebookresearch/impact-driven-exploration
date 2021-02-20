from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class Room:
    def __init__(self,
        top,
        size,
        entryDoorPos,
        exitDoorPos
    ):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos

class MultiRoomNoisyTVEnv(MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10
    ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize

        self.rooms = []

        super(MultiRoomNoisyTVEnv, self).__init__(
            grid_size=25,
            max_steps=self.maxNumRooms * 20
        )

    def _gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # Create the noisy-tv: a ball of arbitrary color that \
            # the agent can change with some action
            if idx == 0:
                self.noisy_tv = Ball(self._rand_elem(COLOR_NAMES))
                self.place_obj(
                    self.noisy_tv, 
                    top=room.top, 
                    size=room.size, 
                    max_tries=100,
                )

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                self.grid.set(*room.entryDoorPos, entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = 'traverse the rooms to get to the goal'

    def _placeRoom(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz+1)
        sizeY = self._rand_int(minSz, maxSz+1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(Room(
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pickup an object -- here the agent should indeed pickup the ball if in front of a door
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # NOTE: there is no Drop action in this case
        # Instead, this action is used to randomly change the color of the noisy-tv 
        elif action == self.actions.drop:
            self.noisy_tv.color = self._rand_elem(COLOR_NAMES)

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

class MultiRoomNoisyTVEnvN7S4(MultiRoomNoisyTVEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=7,
            maxNumRooms=7,
            maxRoomSize=4
        )
        
register(
    id='MiniGrid-MultiRoomNoisyTV-N7-S4-v0',
    entry_point='gym_minigrid.envs:MultiRoomNoisyTVEnvN7S4'
)