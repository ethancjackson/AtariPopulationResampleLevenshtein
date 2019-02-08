from Individual import Individual, from_JSON
from Mutator import mutate
from numpy.random import randint, RandomState
from NetworkMapper import mutate_conv_module
from joblib import Parallel, delayed
import numpy as np
import KerasEvaluator
from numpy import fmax
import KerasConvModules
import gc
import sys
from itertools import product

"""
Novelty. Select top T novelty scores from each generation.
Non-hybrid. Single form of selection pressure. 
Select top-scoring individual in validation among top T novel as elite.
"""

num_cores = 16
game_name = 'Breakout-v0'
exp_name = 'Breakout-GameScore'

def generate_individual(in_shape, out_shape, init_connect_rate, init_seed):
    return Individual(in_shape=in_shape, out_shape=out_shape, init_seed=init_seed, init_connect_rate=init_connect_rate)


def generate_init_population(in_shape, out_shape, num_individuals=10, init_connect_rate=0.5):
    population = []
    for i in range(num_individuals):
        # init_seed = randint(low=0, high=4294967295)
        ind = generate_individual(in_shape, out_shape, init_connect_rate, i)
        population.append(ind)
    return population



def play_atari(individual, game_name, game_iterations, env_seeds, vis=False, sleep=0.05):
    """
    Given an indivudual, play an Atari game and return the score.
    :param individual:
    :return: sequence of actions as numpy array
    """

    from cv2 import cvtColor, COLOR_RGB2GRAY, resize
    import keras as K
    import gym.spaces

    # Initialize Keras model
    conv_module = KerasConvModules.gen_DQN_architecture(input_shape=individual.in_shape,
                                                        output_size=individual.out_size,
                                                        seed=individual.init_seed)

    mutate_conv_module(individual, conv_module, mut_power=0.002)
    me = KerasEvaluator.KerasEvaluator(conv_module)

    # Integer RNG for initial game actions
    rnd_int = RandomState(seed=individual.init_seed).randint

    total_reward = 0
    episode_actions_performed = []
    env = gym.make(game_name)
    action_space_size = env.action_space.n

    for env_seed in env_seeds:

        actions_performed = []
        env.seed(env_seed)
        x = resize(env.reset(), (84, 84))
        x_lum = cvtColor(x, COLOR_RGB2GRAY).reshape(84, 84, 1)
        x = np.concatenate((x, x_lum), axis=2)
        x = x.reshape(1, 84, 84, 4)
        prev_frame = np.zeros_like(x)

        for _ in range(30):
            action = rnd_int(low=0, high=action_space_size)
            env.step(action)
            actions_performed.append(action)

        for i in range(game_iterations):

            # Evaluate network
            step = me.eval(fmax(x, prev_frame))

            # Store frame for reuse
            prev_frame = x

            # Compute one-hot boolean action - THIS WILL DEPEND ON THE GAME
            y_ = step.flatten()
            a = y_.argmax()
            actions_performed.append(a)
            observation, reward, done, info = env.step(a)

            if done:
                # Ensure all action sequences have the same length
                extra_actions = game_iterations - i - 1
                for _ in range(extra_actions):
                    actions_performed.append('x')
                break
            total_reward += reward

            # Use previous frame information to avoid flickering
            x = resize(observation, (84, 84))
            x_lum = cvtColor(x, COLOR_RGB2GRAY).reshape(84, 84, 1)
            x = np.concatenate((x, x_lum), axis=2)
            x = x.reshape(1, 84, 84, 4)

            # sys.stdout.write("Game iteration:{}\tAction:{}\r".format(i, a))
            # sys.stdout.flush()
        episode_actions_performed.append(actions_performed)

    env.close()
    K.backend.clear_session()
    del me

    print('{}{}\n'.format(individual, total_reward))
    return (individual, total_reward, episode_actions_performed)

def partition(episode_archive):
    num_partitions = len(episode_archive) if len(episode_archive) < num_cores else num_cores
    archive_partitions = []
    items_per_partition = len(episode_archive) // num_partitions
    for i in range(num_partitions):
        archive_partitions.append(episode_archive[i * items_per_partition:(i + 1) * items_per_partition])
    return archive_partitions


def save_population(population, generation, expname):
    import pickle, gzip
    population_json = [ind.to_JSON() for ind in population]
    file = gzip.open('./{}/population-{}-{}.pkl'.format(exp_name, expname, generation), 'wb')
    pickle.dump(population_json, file)
    file.flush()
    file.close()

def load_population(filename):
    import pickle, gzip
    with gzip.open(filename, 'rb') as file:
        population_json = pickle.load(file)
        population = [from_JSON(ind) for ind in population_json]
        generations = len(population[len(population)-1].generation_seeds)
        file.close()
        return population, generations



def evolve_solution(game_name, action_space_size, pop_size=1000+1, n_generations=1000,
                    training_frames=20000, training_episodes=(0,), starting_generation=0, T=40, population=None,
                    validation_frames=10000, validation_episodes=(1,)):

    in_shape = (84,84,4)
    out_shape = (action_space_size,)
    init_connect_rate = 1.0

    num_episodes = len(training_episodes)

    if population is None:
        population = generate_init_population(in_shape, out_shape, pop_size, init_connect_rate)


    rnd_int = np.random.RandomState(seed=1).randint
    rnd_uniform = np.random.RandomState(seed=1).uniform

    mean_game_scores = []
    best_game_scores = []
    mean_validation_scores = []

    for gen in range(starting_generation, n_generations):

        print('\nGeneration', gen)
        save_population(population, gen, exp_name)

        # PARALLEL
        results = Parallel(n_jobs=num_cores)(
            delayed(play_atari)(
                ind, game_name, training_frames, training_episodes) for ind in population
        )

        # Retrieve game scores
        game_scores = [x[1] for x in results]

        # Compute mean game score
        mean_game_score = np.mean(game_scores)
        mean_game_scores.append(mean_game_score)

        # Find top-scoring individual
        results = sorted(results, key=lambda x: x[1])
        g_elite_ind, g_elite_game_score, g_elite_actions_performed = results[-1]

        best_game_scores.append(g_elite_game_score)

        file = open('./{}/{}-elite_game-{}-score-{}.txt'.format(exp_name, game_name, gen, g_elite_game_score), 'w')
        file.write(g_elite_ind.to_JSON())
        file.flush()
        file.close()

        print('Best Score (Game Score): {} {:.2f}'.format(g_elite_ind, g_elite_game_score))
        print('Previous Mean Game Scores: {}'.format(mean_game_scores))
        print('Current Mean Game Score: {:.2f}'.format(mean_game_score))
        print('Previous Best Game Scores: {}'.format(best_game_scores))
        print('Current Best Game Score: {}'.format(best_game_scores[-1]))

        # Truncate based on novelty score
        pop_trunc = [result for result in results[-T:]]

        # Then select half with highest game score (Not in main experiment)
        # pop_trunc = sorted(pop_trunc, key=lambda x: x[1])[-T // 2:]

        # Use cross-validation to select elite
        validation_pop = [result[0] for result in pop_trunc[-10:]]
        print([x.init_seed for x in validation_pop])
        validation_runs = product(validation_pop, validation_episodes)
        # print([(x.init_seed, y) for x,y in validation_runs])

        print('Running Validation Episodes.')
        validation_results = Parallel(n_jobs=num_cores)(
            delayed(play_atari)(
                ind, game_name, validation_frames, [episode]) for ind, episode in validation_runs
        )

        # Collect scores by individual
        validation_scores = []
        for ind in validation_pop:
            validation_scores.append((ind, np.mean([v_score for v_ind, v_score, _ in validation_results if v_ind == ind])))
        validation_scores = sorted(validation_scores, key=lambda x: x[1])

        # Sort based on game score and select top individual
        top_validation_ind, top_validation_score = validation_scores[-1]

        # Add best-generalizing individual to reproduction population
        # pop_trunc.append((top_validation_ind, None, None, None))

        mean_validation_scores.append(top_validation_score)
        print('Previous Mean Validation Scores: {}'.format(mean_validation_scores))
        print('Current Mean Validation Score Over {} Episodes: {:.2f}'.format(len(validation_episodes), mean_validation_scores[-1]))

        file = open('./{}/{}-elite_validation-{}-mean_score-{}.txt'.format(exp_name, game_name, gen, mean_validation_scores[-1]), 'w')
        file.write(top_validation_ind.to_JSON())
        file.flush()
        file.close()

        run_file = open('./{}/run_info.txt'.format(exp_name), 'w')
        run_file.write('Mean Game Scores\n{}\n'.format(mean_game_scores))
        run_file.write('Best Game Scores\n{}\n'.format(best_game_scores))
        run_file.write('Mean Validation Scores Over {} Episodes\n{}\n'.format(len(validation_episodes), mean_validation_scores))
        run_file.flush()
        run_file.close()

        if gen == n_generations - 1:
            return g_elite_game_score
        # Initialize population with elite ind
        new_pop = [top_validation_ind]
        for _ in range(pop_size - 1):
            offspring, _, _ = pop_trunc[rnd_int(low=0, high=len(pop_trunc))]
            offspring = offspring.copy()
            mutate(offspring)
            new_pop.append(offspring)

        population = new_pop
        gc.collect()


### Testing
import gym.spaces
env = gym.make(game_name)
action_space_size = env.action_space.n
env.close()
pop, _ = load_population('./Breakout-GameScore/population-Breakout-GameScore-193.pkl')
evolve_solution(game_name, action_space_size, pop_size=1000+1,
                training_episodes=[0], T=20, starting_generation=193, population=pop,
                training_frames=5000, validation_frames=5000, validation_episodes=list(range(1,31)))
