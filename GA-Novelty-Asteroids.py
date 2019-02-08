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
from itertools import product, chain
import Levenshtein

"""
Novelty. Select top T novelty scores from each generation.
Non-hybrid. Single form of selection pressure. 
Select top-scoring individual in validation among top T novel as elite.
"""

num_cores = 16
game_name = 'Asteroids-v0'
exp_name = 'Asteroids-Archive'

def generate_individual(in_shape, out_shape, init_connect_rate, init_seed):
    return Individual(in_shape=in_shape, out_shape=out_shape, init_seed=init_seed, init_connect_rate=init_connect_rate)


def generate_init_population(in_shape, out_shape, num_individuals=10, init_connect_rate=1.0):
    population = []
    for i in range(num_individuals):
        # init_seed = randint(low=0, high=4294967295)
        ind = generate_individual(in_shape, out_shape, init_connect_rate, i)
        population.append(ind)
    return population

def levenshtein_distance(seq1, seq2):
    min_len = np.min((len(seq1), len(seq2)))
    # print(seq1)
    # print(seq2)
    return Levenshtein.distance(seq1[:min_len], seq2[:min_len])

def play_atari(individual, game_name, game_iterations, env_seeds, mut_rate=0.002, vis=False, sleep=0.05):
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

    mutate_conv_module(individual, conv_module, mut_power=mut_rate)
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
            # actions_performed.append(action)

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
                # extra_actions = game_iterations - i - 1
                # for _ in range(extra_actions):
                #     actions_performed.append('x')
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

    sys.stdout.write('{}{}\n'.format(individual, total_reward))
    sys.stdout.flush()
    return (individual, total_reward, episode_actions_performed)

def partition(archive, num_partitions=num_cores):
    archive_partitions = []
    items_per_partition = len(archive) // num_partitions
    for i in range(num_partitions):
        archive_partitions.append(archive[i*items_per_partition:(i+1)*items_per_partition])
    return archive_partitions

def compute_distances_segmented(seq, archive, segment_size):
    num_segments = len(seq) // segment_size + 1
    segment_distances = []
    for i in range(num_segments):
        segment_distances.append([levenshtein_distance(seq[i*segment_size:(i+1)*segment_size], x[i*segment_size:(i+1)*segment_size]) for x in archive])
    distances = np.array(segment_distances)
    sum_distances = np.sum(distances, axis=0)
    return sum_distances.tolist()

def compute_novelty(individual, game_score, actions_performed, archive, segment_size=500, n_neighbours=25):

    num_episodes = len(actions_performed)

    # PARALLEL
    novelty_scores = []
    for episode in range(num_episodes):
        # convert actions_performed to a string
        episode_actions_performed = ''.join(str(i) for i in actions_performed[episode])
        # print(episode_actions_performed)
        # print()
        archive_partitions = partition(archive[episode])
        distance_lists = Parallel(n_jobs=num_cores)(
            delayed(compute_distances_segmented)(
                episode_actions_performed, part, segment_size
            ) for part in archive_partitions
        )

        # compute distance between actions performed and sequences in archive
        distances = list(chain(*distance_lists))

        # find distances to n_neighbours
        nearest_neighbours = sorted(distances)[:n_neighbours]

        del distance_lists
        del distances

        novelty_scores.append(np.mean(nearest_neighbours))
    # print(novelty_scores)
    novelty_score = np.mean(novelty_scores)
    # return (individual, game_score, actions_performed, mean distance to nearest neighbours)
    # sys.stdout.write('{} {} {}\n'.format(individual, game_score, novelty_score))
    # sys.stdout.flush()
    return (individual, game_score, actions_performed, novelty_score)

def population_novelty(population, archive, return_n):
    """
    Given a population and an archive, return the archived individuals whose behaviours were most different
    from those in population. This should force the search to consider individuals that will produce different
    behaviours and results than the current population.
    :param population:
    :param archive:
    :return:
    """
    results = []
    sequences_only = []

    pop_as_archive = [seq[0] for ind, seq in population]
    for seq in pop_as_archive:
        sequences_only.append(''.join(str(i) for i in seq))

    print('Computing {} scores.'.format(len(archive)))
    for ind, actions_performed, gs in archive:
        # Take the first F actions only since archived records may contain fewer actions than in current pop
        # sequences_trunc = [seq[:len(actions_performed)] for seq in sequences_only]
        # print(actions_performed)
        # print([sequences_only])
        results.append(compute_novelty(ind, gs, actions_performed, [sequences_only]))
        print(ind, gs, results[-1][3])
    # sort results by novelty score
    results = sorted(results, key=lambda x: x[3])
    # for r in results:
    #     print(r[0], r[1], r[3])
    return [ind for ind, _, _, _ in results][-return_n:]


def save_population(population, generation, expname):
    import pickle, gzip
    population_list = [ind for ind in population]
    file = gzip.open('./{}/population-{}-{}.pkl'.format(exp_name, expname, generation), 'wb')
    pickle.dump((generation, population_list), file)
    file.flush()
    file.close()

def load_population(filename):
    import pickle, gzip
    with gzip.open(filename, 'rb') as file:
        generations, population = pickle.load(file)
        return population, generations

def save_archive(archive, gen, expname):
    import pickle, gzip
    file = gzip.open('./{}/archive-{}.pkl'.format(exp_name, expname), 'wb')
    pickle.dump(archive, file)
    file.flush()
    file.close()

def load_archive(filename):
    import pickle, gzip
    with gzip.open(filename, 'rb') as file:
        archive = pickle.load(file)
        return archive


def evolve_solution(game_name, action_space_size, pop_size=1000+1, archive_p=0.1, archive_dump=None, n_generations=5,
                    training_frames=20000, training_episodes=(0,), starting_generation=0, T=40, F=200, archive=None, population=None,
                    validation_frames=10000, validation_episodes=(1,), validation_archive=None, validation_archive_p=1.0,
                    improvement_cutoff=5):


    if population is None:
        population = generate_init_population(num_individuals=pop_size, in_shape=(84,84,4), out_shape=(action_space_size,))

    rnd_int = np.random.randint

    if archive is None:
        archive = []

    mean_training_game_scores = []
    best_training_game_scores = []
    best_validation_game_scores = []
    rand = np.random.RandomState(seed=1).uniform

    improvement_gens = 0

    for gen in range(starting_generation, n_generations):

        save_population(population, gen, exp_name)
        improvement_gens += 1

        # PARALLEL
        results = Parallel(n_jobs=num_cores)(
            delayed(play_atari)(
                ind, game_name, training_frames, training_episodes, 0.002) for ind in population
        )

        results = sorted(results, key=lambda x: x[1])

        for ind, gs, actions_performed in results:
            if rand() < archive_p:
                archive.append((ind, actions_performed, gs))

        save_archive(archive, gen, exp_name)

        results_trunc = [result for result in results[-T:]]
        parents = [ind for ind, _, _ in results_trunc]

        # Use cross-validation to select elite
        validation_pop = [result[0] for result in results_trunc[-10:]]
        validation_runs = product(validation_pop, validation_episodes)
        # print([(x.init_seed, y) for x,y in validation_runs])

        print('Running Validation Episodes.')
        validation_results = Parallel(n_jobs=num_cores)(
            delayed(play_atari)(
                ind, game_name, validation_frames, [episode], 0.002) for ind, episode in validation_runs
        )

        # Collect scores by individual
        validation_scores = []
        for ind in validation_pop:
            validation_scores.append(
                (ind, np.mean([v_score for v_ind, v_score, _ in validation_results if v_ind == ind])))
        validation_scores = sorted(validation_scores, key=lambda x: x[1])

        # Sort based on game score and select top individual
        top_validation_ind, top_validation_score = validation_scores[-1]

        # Add best-generalizing individual to reproduction population
        # pop_trunc.append((top_validation_ind, None, None, None))

        best_validation_game_scores.append(top_validation_score)
        print('Previous Mean Validation Scores: {}'.format(best_validation_game_scores))
        print('Current Mean Validation Score Over {} Episodes: {:.2f}'.format(len(validation_episodes),
                                                                              best_validation_game_scores[-1]))

        file = open('./{}/{}-elite_validation-gen-{}-score-{}.txt'.format(exp_name, game_name, gen,
                                                                           best_validation_game_scores[-1]), 'w')
        file.write(top_validation_ind.to_JSON())
        file.flush()
        file.close()

        # Store top training game score
        best_ind, best_score, best_actions = results[-1]
        best_training_game_scores.append(best_score)
        mean_training_game_scores.append(np.mean([score for _, score, _ in results]))

        # if top_validation_score > best_score:
        #     best_score = top_validation_score

        print('Mean Training Score: {}'.format(mean_training_game_scores[-1]))
        print('Best Training Score: {}'.format( best_training_game_scores[-1]))

        file = open('./{}/{}-best_training-gen-{}-score-{}.txt'.format(exp_name, game_name, gen, best_training_game_scores[-1]), 'w')
        file.write(best_ind.to_JSON())
        file.flush()
        file.close()

        run_file = open('./{}/run_info.txt'.format(exp_name), 'w')
        run_file.write('Mean Game Scores\n{}\n'.format(mean_training_game_scores))
        run_file.write('Best Game Scores\n{}\n'.format(best_training_game_scores))
        run_file.write('Best Validation Scores (Best Mean Over {} Episodes)\n{}\n'.format(len(validation_episodes),
                                                                              best_validation_game_scores))

        run_file.flush()
        run_file.close()

        new_pop = [top_validation_ind]
        for _ in range(pop_size - 1):
            offspring = parents[rnd_int(low=0, high=len(parents))]
            offspring = offspring.copy()
            mutate(offspring)
            new_pop.append(offspring)
        population = new_pop


        # Find Best Scoring Individual and Phase Mean
        # best_ind, best_score, best_actions = results[-1]

        if improvement_gens >= improvement_cutoff:

            # Get the latest X validation scores
            recent_validation_scores = best_validation_game_scores[-improvement_cutoff:]
            differences = [x - recent_validation_scores[0] for x in recent_validation_scores]

            print(differences)

            # Has there been no progress over the last X generations?
            if np.all([x <= 0 for x in differences]):
                # Reset improvement_gens to give new pop a chance
                improvement_gens = 0
                # Don't just continue, resample population from archive
                print('Computing novelty scores between population and archive.')
                current_population = [(ind, actions) for ind, score, actions in results]
                parents = population_novelty(current_population, archive, return_n=T)
                new_pop = parents
                for _ in range(pop_size - len(new_pop)):
                    offspring = parents[rnd_int(low=0, high=len(parents))]
                    offspring = offspring.copy()
                    mutate(offspring)
                    new_pop.append(offspring)
                population = new_pop



# ### Testing
import gym.spaces
env = gym.make(game_name)
action_space_size = env.action_space.n
env.close()
evolve_solution(game_name, action_space_size, pop_size=1000+1, archive_p=0.01, archive_dump=None,
                training_episodes=[0], T=20, n_generations=1000, improvement_cutoff=10, archive=None, population=None,
                training_frames=20000, validation_frames=20000, validation_episodes=list(range(1,31)))
