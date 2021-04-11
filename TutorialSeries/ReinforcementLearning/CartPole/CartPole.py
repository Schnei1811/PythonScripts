import numpy as np
import tensorflow as tf
import gym

def resetGradBuffer(gradBuffer):
    for ix, grad in enumerate(gradBuffer): gradBuffer[ix] = grad * 0
    return gradBuffer

def discount_rewards(r):
    # take 1D float array of rewards and compute discounted reward
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# This function uses our model to produce a new state when given a previous state and action
def stepModel(sess, xs, action):
    toFeed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1, 5])
    myPredict = sess.run([predicted_state], feed_dict={previous_state: toFeed})
    reward = myPredict[0][:, 4]
    observation = myPredict[0][:, 0:4]
    observation[:, 0] = np.clip(observation[:, 0], -2.4, 2.4)
    observation[:, 2] = np.clip(observation[:, 2], -0.4, 0.4)
    doneP = np.clip(myPredict[0][:, 5], 0, 1)
    if doneP > 0.1 or len(xs) >= 300: done = True
    else: done = False
    return observation, reward, done

resume = False                  # resume from previous checkpoint?
env = gym.make('CartPole-v0')

tf.reset_default_graph()

# hyperparameters
num_hidden_layers = 8           # number of hidden layer neurons
learning_rate = 1e-2
gamma = 0.99                    # discount factor for reward
decay_rate = 0.99               # decay factor for RMSProp    leaky sum of grad^2
model_layer_size = 256          # model layer size
num_observations = len(env.reset())

model_batch_size = 3            # Batch size when learning from model
real_batch_size = 3             # Batch size when learning from real environment

#initiaize variables
observations = tf.placeholder(tf.float32, [None, num_observations], name="input_x")        # 2 observations
input_data = tf.placeholder(tf.float32, [None, 5])
input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")             # y
reward_signal = tf.placeholder(tf.float32, name="reward_signal")            # reward
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")                     # weights grad 1
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")                     # weights grad 2
batchGrad = [W1Grad, W2Grad]                                                # collective weights

# initialize layers
# xavier initialization: a form of initialization where scale of the gradients roughly the same in all layers
W1 = tf.get_variable("W1", shape=[num_observations, num_hidden_layers], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[num_hidden_layers, 1], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
score = tf.matmul(layer1, W2)               # continuous Data
probability = tf.nn.sigmoid(score)          # discrete

# useful functions
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))    #log likelihood
loss = -tf.reduce_mean(loglik * reward_signal)              # loss function: -log likelihood

# tf.gradients() Constructs symbolic partial derivatives of sum of ys w.r.t. x in xs

training_variables = tf.trainable_variables()
newGrads = tf.gradients(loss, training_variables)
updateGrads = tf.train.AdamOptimizer(learning_rate=learning_rate).apply_gradients(zip(batchGrad, training_variables))

# creates a tag rnnlm with softmax_w and softmax_b
with tf.variable_scope('rnnlm'):
    softmax_b = tf.get_variable("softmax_b", [50])
    softmax_w = tf.get_variable("softmax_w", [model_layer_size, 50])

# initialize model state layer 1 and 2
previous_state = tf.placeholder(tf.float32, [None, 5], name="previous_state")
B1M = tf.Variable(tf.zeros([model_layer_size]), name="B1M")     #bias first layer
W1M = tf.get_variable("W1M", shape=[5, model_layer_size], initializer=tf.contrib.layers.xavier_initializer())
layer1M = tf.nn.relu(tf.matmul(previous_state, W1M) + B1M)

B2M = tf.Variable(tf.zeros([model_layer_size]), name="B2M")
W2M = tf.get_variable("W2M", shape=[model_layer_size, model_layer_size], initializer=tf.contrib.layers.xavier_initializer())
layer2M = tf.nn.relu(tf.matmul(layer1M, W2M) + B2M)

# observations bias and weights
bObsv = tf.Variable(tf.zeros([num_observations]), name="bO")
wObsv = tf.get_variable("wO", shape=[model_layer_size, num_observations], initializer=tf.contrib.layers.xavier_initializer())

# reward
bRew = tf.Variable(tf.zeros([1]), name="bR")
wRew = tf.get_variable("wR", shape=[model_layer_size, 1], initializer=tf.contrib.layers.xavier_initializer())

# Complete
bCom = tf.Variable(tf.ones([1]), name="bD")
wCom = tf.get_variable("wD", shape=[model_layer_size, 1], initializer=tf.contrib.layers.xavier_initializer())

predicted_observation = tf.matmul(layer2M, wObsv, name="predicted_observation") + bObsv
predicted_reward = tf.matmul(layer2M, wRew, name="predicted_reward") + bRew
predicted_completion = tf.sigmoid(tf.matmul(layer2M, wCom, name="predicted_done") + bCom)

true_observation = tf.placeholder(tf.float32, [None, num_observations], name="true_observation")
true_reward = tf.placeholder(tf.float32, [None, 1], name="true_reward")
true_completion = tf.placeholder(tf.float32, [None, 1], name="true_done")

# amalgamation of states
predicted_state = tf.concat([predicted_observation, predicted_reward, predicted_completion], 1)

# loss functions
observation_loss = tf.square(true_observation - predicted_observation)
reward_loss = tf.square(true_reward - predicted_reward)
completion_loss = tf.multiply(predicted_completion, true_completion) + \
                  tf.multiply(1 - predicted_completion, 1 - true_completion)
completion_loss = -tf.log(completion_loss)

# total model loss
model_loss = tf.reduce_mean(observation_loss + reward_loss + completion_loss)

# Minimize model loss using adam optimizer
updateModel = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model_loss)

xs, drs, ys, ds = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1
switch_point = 1
init = tf.global_variables_initializer()
batch_size = real_batch_size

drawFromModel = False       # When set to True, will use model for observations
trainTheModel = True        # Whether to train the model
trainThePolicy = False      # Whether to train the policy

print('Num Observations {}'.format(num_observations))
print(env.reset())
print(env.reset().shape)

# Launch the graph
with tf.Session() as sess:
    rendering = True
    sess.run(init)
    observation = env.reset()
    x = observation
    gradBuffer = sess.run(training_variables)
    gradBuffer = resetGradBuffer(gradBuffer)

    while episode_number <= 5000:
        # Start displaying environment once performance is acceptably high.

        if (reward_sum / batch_size > 150 and drawFromModel == False) or rendering == True:
            env.render()
            rendering = True

        x = np.reshape(observation, [1, num_observations])

        tfprob = sess.run(probability, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0

        # record various intermediates (needed later for backprop)
        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)

        # step the  model or real environment and get new measurements
        if drawFromModel == False: observation, reward, done, info = env.step(action)
        else: observation, reward, done = stepModel(sess, xs, action)

        reward_sum += reward

        ds.append(done * 1)
        drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

        if done:
            if drawFromModel == False: real_episodes += 1
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            epd = np.vstack(ds)
            xs, drs, ys, ds = [], [], [], []  # reset array memory

            if trainTheModel == True:
                actions = np.array([np.abs(y - 1) for y in epy][:-1])
                state_prevs = epx[:-1, :]
                state_prevs = np.hstack([state_prevs, actions])
                state_nexts = epx[1:, :]
                rewards = np.array(epr[1:, :])
                print(rewards)
                dones = np.array(epd[1:, :])
                state_nextsAll = np.hstack([state_nexts, rewards, dones])

                feed_dict = {previous_state: state_prevs, true_observation: state_nexts, true_completion: dones,
                             true_reward: rewards}
                loss, pState, _ = sess.run([model_loss, predicted_state, updateModel], feed_dict)
            if trainThePolicy == True:
                discounted_epr = discount_rewards(epr).astype('float32')
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, reward_signal: discounted_epr})

                # If gradients become too large, end training process
                if np.sum(tGrad[0] == tGrad[0]) == 0: break
                for ix, grad in enumerate(tGrad): gradBuffer[ix] += grad

            if switch_point + batch_size == episode_number:
                switch_point = episode_number
                if trainThePolicy == True:
                    sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                    gradBuffer = resetGradBuffer(gradBuffer)

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

                if drawFromModel == False:
                    print('World Perf: Episode %f. Reward %f. action: %f. mean reward %f.' % (
                    real_episodes, reward_sum / real_batch_size, action, running_reward / real_batch_size))
                    if reward_sum / batch_size > 200: break
                reward_sum = 0

                # Once the model has been trained on 1xdif-100 episodes, we start alternating between training the policy
                # from the model and training the model from the real environment.
                if episode_number > 100:
                    drawFromModel = not drawFromModel
                    trainTheModel = not trainTheModel
                    trainThePolicy = not trainThePolicy

            if drawFromModel == True:
                observation = np.random.uniform(-0.1, 0.1, [4])  # Generate reasonable starting point
                batch_size = model_batch_size
            else:
                observation = env.reset()
                batch_size = real_batch_size

print(real_episodes)


