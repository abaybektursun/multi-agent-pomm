import pommerman

from fc_agent import FCAgent

import numpy as np
import pommerman.agents  as agents
import matplotlib.pyplot as plt

from collections import namedtuple

def main(save_img=False):
    # Print all possible environments in the Pommerman registry
    print(pommerman.registry)

    # Create a set of agents (exactly four)
    agent_list = [
        FCAgent(),
        agents.RandomAgent(),
        agents.SimpleAgent(),
        agents.RandomAgent(),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFA-v0', agent_list)

    ax1 = plt.subplot(331)
    ax2 = plt.subplot(332)
    ax3 = plt.subplot(333)
    ax4 = plt.subplot(334)
    ax5 = plt.subplot(335)
    ax6 = plt.subplot(336)
    ax7 = plt.subplot(337)
    ax8 = plt.subplot(338)
    ax9 = plt.subplot(339)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        step = 0
        while not done:
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            step += 1

            if save_img == True:
                # Plot
                ax1.imshow(state[0]['board'])
                ax2.imshow(state[0]['bomb_blast_strength'])
                ax3.imshow(state[0]['bomb_life'])
                ax4.imshow(state[1]['board'])
                ax5.imshow(state[1]['bomb_blast_strength'])
                ax6.imshow(state[1]['bomb_life'])
                ax7.imshow(state[2]['board'])
                ax8.imshow(state[2]['bomb_blast_strength'])
                ax9.imshow(state[2]['bomb_life'])
                plt.savefig('frames/{}.png'.format(step))

        print("# Steps: ", step)
        print('Episode {} finished'.format(i_episode))
        # Print the result
        print(info)
    env.close()


if __name__ == '__main__':
    main(save_img=False)
