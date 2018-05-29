import numpy as np

class _utils:
    def __init__(self, board_h, board_w, save_file):
        self.save_file = save_file
        self.num_actions = 6
        self.board_area = board_h * board_w

        self.int2vec = {
            1  : np.zeros((self.board_area,)),
            2  : np.zeros((self.board_area,)),
            4  : np.zeros((self.board_area,)),
            6  : np.zeros((self.board_area,)),
            7  : np.zeros((self.board_area,)),
            8  : np.zeros((self.board_area,)),
            10 : np.zeros((self.board_area,)),
            11 : np.zeros((self.board_area,)),
            12 : np.zeros((self.board_area,)),
            13 : np.zeros((self.board_area,))
        }
        self.blast_strength_vec = np.zeros((max(board_h, board_w)+1,))

        self.max_ammo = 4
        self.ammo = np.zeros((self.max_ammo,))

        self.this_agent = np.zeros((5,))
        self.friend = np.zeros((5,))
        self.enemy1 = np.zeros((5,))
        self.enemy2 = np.zeros((5,))
        self.enemy3 = np.zeros((5,))

        # Different symbolic objects
        self.input_size = self.board_area*len(self.int2vec) + \
            max(board_h, board_w)+1 + \
            self.max_ammo + \
            5*5 + \
            self.board_area + \
            self.board_area
        # Action and reward

    def input(self, obs):
        blast_strength = int(obs['blast_strength'])
        ammo        = int(obs['ammo'])
        my_position = tuple(obs['position'])
        teammate    = int(obs['teammate'].value) - 9
        enemies     = np.array([e.value for e in obs['enemies']]) - 9
        board       = np.array(obs['board'])
        bombs       = np.array(obs['bomb_blast_strength'])/2.0
        bombs_life  = np.array(obs['bomb_life'])/9.0

        # Symbolic objects to vector of boards
        for idx, cell in enumerate(board.flatten().tolist()):
            if cell in self.int2vec:
                self.int2vec[cell][idx] = 1.0

        # !TODO Test this assumption
        self.blast_strength_vec[blast_strength] = 1.0

        # If ammo > 10, ammo = 10 (as one hot)
        self.ammo[min(self.max_ammo,ammo)-1] = 1.0

        agent_ids = [0,1,2,3,4]
        # Agents
        for an_enemy_id, an_enemy_vec in zip(enemies, [self.enemy1, self.enemy2, self.enemy3]):
            an_enemy_vec[an_enemy_id] = 1.0
            agent_ids.remove(an_enemy_id)
        self.friend[teammate] = 1.0
        agent_ids.remove(teammate)
        # DEBUG
        if len(agent_ids) != 1: raise ValueError('Error! agent_ids has more/less than one id left!')
        # DEBUG
        self.this_agent[agent_ids[0]] = 1.0


        # !TODO Concatenate all the vectors
        input_data = np.array([])
        for idx in self.int2vec:
            input_data = np.concatenate((input_data, self.int2vec[idx]))

        input_data = np.concatenate((input_data, self.blast_strength_vec))
        input_data = np.concatenate((input_data, self.ammo))
        input_data = np.concatenate((input_data, self.this_agent))
        input_data = np.concatenate((input_data, self.friend))
        input_data = np.concatenate((input_data, self.enemy1))
        input_data = np.concatenate((input_data, self.enemy2))
        input_data = np.concatenate((input_data, self.enemy3))
        input_data = np.concatenate((input_data, bombs.flatten()))
        input_data = np.concatenate((input_data, bombs_life.flatten()))

        #print("Data vector: {} v.s. input_size: {}".format(input_data.shape, self.input_size))

        return input_data.flatten()

    def action_onehot(self, action):
        action_vec = np.zeros((self.num_actions,))
        action_vec[action] = 1.0
        return action_vec

    def save_torch(self, model):
        torch.save(model, self.save_file)

