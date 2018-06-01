# Training
import pommerman

import tensorflow as tf
import tarfile

from pommerman.runner import DockerAgentRunner

from rnn_agent import RNN_Agent   

class MyAgent(DockerAgentRunner):
    def __init__(self, checkpoint):
        self._agent = RNN_Agent(model_training='')
        saver = tf.train.Saver()
        latest_model = tf.train.latest_checkpoint(checkpoint)
        if latest_model is not None:
            saver.restore(
                rnn_agent.sess,
                latest_model
            )
            print("Restored ", latest_model)
        else: 
            print('Unable to load the pre-trained model, please contact me: abaybektursun@gmail.com')
            exit(0)
        
    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)


if __name__ == '__main__':
    model = 'model'
    
    agent = MyAgent('models/'+model+'/' )
    agent.run()
    
    
    #agent._agent.sess.close()

     
