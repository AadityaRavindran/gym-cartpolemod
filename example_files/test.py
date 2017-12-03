import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt
from collections import deque
import gym_cartpolemod

TIME_STEPS = 600000
TRIALS = 1000
RUNS = 100
success_score = 6000

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters",[4,2])
        state = tf.placeholder("float",[None,4])
        actions = tf.placeholder("float",[None,2])
        advantages = tf.placeholder("float",[None,1])
        linear = tf.matmul(state,params)
        probabilities = tf.nn.softmax(linear)
        good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions),reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probabilities, state, actions, advantages, optimizer

def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float",[None,4])
        newvals = tf.placeholder("float",[None,1])
        w1 = tf.get_variable("w1",[4,10])
        b1 = tf.get_variable("b1",[10])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[10,1])
        b2 = tf.get_variable("b2",[1])
        calculated = tf.matmul(h1,w2) + b2
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss

def run_episode(env, policy_grad, value_grad, sess):
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []


    for _ in range(TIME_STEPS):
        # env.render()
        # calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calculated,feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0,1) < probs[0][0] else 1
        # record the transition
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in range(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.97
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calculated,feed_dict={vl_state: obs_vector})[0][0]

        # advantage: how much better was this action than normal
        advantages.append(future_reward - currentval)

        # update the value function towards new return
        update_vals.append(future_reward)

    # update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

    return totalreward


env = gym.make('CartPoleMod-v0')
env._max_episode_steps = TIME_STEPS
policy_grad = policy_gradient()
value_grad = value_gradient()
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

trial_score = deque(maxlen = RUNS)
total_success = 0
for run in range(1,RUNS+1):
    scores = deque(maxlen = 20)
    success = 0
    for trial in range(1,TRIALS+1):
        total_reward = 0
        reward = run_episode(env, policy_grad, value_grad, sess)
        scores.append(reward)
        if trial%100==0: print('Trial: {} reward: {}'.format(trial,reward))
        try:
            mean_score = np.mean(scores)
        except:
            mean_score = 0
        if mean_score > success_score:
            trial_score.append(trial)
            success = 1
            print('Successful trial. Run#:{} Score: {}'.format(run,mean_score))
            
    mean_trial = np.mean(trial_score)
    total_success += success
    if success==0:
        print('Failed Run#:{}'.format(run))
    else:
        print('Successful run#: {} Average trial#: {}'.format(run,mean_trial))
print('\n\n\n\n\n\nSuccess Rate:{}% #Trials: {}'.format(total_success,mean_trial))
