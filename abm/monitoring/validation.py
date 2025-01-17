from abm.start_sim import start

from pathlib import Path
import pickle
import numpy as np
import multiprocessing as mp
import os
import dotenv as de


def rerun_NNs(name, num_NNs=20, num_seeds=100, noise_type=None, perturb_type=None):

    if noise_type is None:
        print(f'running: {name}')
    else:
        print(f'running: {name} + {noise_type} noise')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/{name}'
    env_path = fr'{exp_path}/.env'
    envconf = de.dotenv_values(env_path)

    if Path(fr'{exp_path}/val_results_cen.txt').is_file():
        return print(f'val_results_cen already exists')
    
    with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
        data = pickle.load(f)
    data_genxpop = np.mean(data, axis=2) # average across episodes
    avg_data = np.mean(data_genxpop, axis=1) # average across individuals in population --> different than rerun_topNNs()

    # --> NN0 / top
    # top_data = np.min(data_genxpop, axis=1) # min : top
    # # top_data = np.max(data_genxpop, axis=1) # max : top
    # top_ind = np.argsort(top_data)[:num_NNs] # min : top
    # # top_ind = np.argsort(top_data)[-1:-num_NNs-1:-1] # max : top
    # top_fit = [top_data[i] for i in top_ind]

    # --> NNcen / dist center
    if envconf['SIM_TYPE'] == 'walls':
        top_ind = np.argsort(avg_data)[:num_NNs] # min : top
    elif envconf['SIM_TYPE'] == 'nowalls':
        top_ind = np.argsort(avg_data)[-1:-num_NNs-1:-1] # max : top
    else:
        raise ValueError('SIM_TYPE not recognized')
    top_fit = [avg_data[i] for i in top_ind]

    val_matrix = np.zeros((num_NNs,
                           num_seeds))
    print(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}')

    # pack inputs for multiprocessing map
    mp_inputs = []
    for g,f in zip(top_ind, top_fit):

        # with open(fr'{data_dir}/{name}/gen{g}_NN0_pickle.bin','rb') as f: # --> NN0 / top
        with open(fr'{data_dir}/{name}/gen{g}_NNcen_pickle.bin','rb') as f: # --> NNcen / dist center
            pv = pickle.load(f)

        for s in range(num_seeds):
            mp_inputs.append( (None, pv, None, s, env_path) ) # model_tuple=None, load_dir=None

    # run agent NNs in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async(start, mp_inputs)
        pool.close()
        pool.join()
    results_list = results.get()

    # skip to start of each seed series/chunk + allocate fitness to save matrix
    if envconf['SIM_TYPE'] == 'walls':
        for i,c in enumerate(range(0, len(results_list), num_seeds)):
            for s,(time_taken, dist_from_patch) in enumerate(results_list[c : c + num_seeds]):
                val_matrix[i,s] = int(time_taken)
    elif envconf['SIM_TYPE'] == 'nowalls':
        for i,c in enumerate(range(0, len(results_list), num_seeds)):
            for s,(time_taken, total_res_collected) in enumerate(results_list[c : c + num_seeds]):
                val_matrix[i,s] = int(total_res_collected)

    # saving protocol for noise/perturb/regular

    if noise_type is not None:

        # dump raw data + calc avg val perf
        with open(fr'{exp_path}/val_matrix_cen_{noise_type}_noise.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        avg_per_NN_new = np.average(val_matrix, axis=1).round(1)

        # previous val perf
        with open(fr'{exp_path}/val_matrix_cen.bin','rb') as f:
            val_matrix = pickle.load(f)
        avg_per_NN_prev = np.average(val_matrix, axis=1).round(1)

        # print both
        for i, ef, vfn, vfp in zip(top_ind, top_fit, avg_per_NN_new, avg_per_NN_prev):
            print(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vfp)} | val_fit + noise: {int(vfn)}')

    elif perturb_type is not None:

        with open(fr'{exp_path}/val_matrix_cen_{perturb_type}_perturb.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        avg_per_NN_new = np.average(val_matrix, axis=1).round(1)

        with open(fr'{exp_path}/val_matrix_cen.bin','rb') as f:
            val_matrix = pickle.load(f)
        avg_per_NN_prev = np.average(val_matrix, axis=1).round(1)

        for i, ef, vfn, vfp in zip(top_ind, top_fit, avg_per_NN_new, avg_per_NN_prev):
            print(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vfp)} | val_fit + {perturb_type}: {int(vfn)}')

    else:

        with open(fr'{exp_path}/val_matrix_cen.bin', 'wb') as f:
            pickle.dump(val_matrix, f)
        avg_per_NN = np.average(val_matrix, axis=1).round(1)

        with open(fr'{exp_path}/val_results_cen.txt', 'w') as f:
            f.write(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}\n')
            for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
                f.write(str(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vf)}\n'))

        for i, ef, vf in zip(top_ind, top_fit, avg_per_NN):
            print(f'gen: {i} | EA_fit: {int(ef)} | val_fit: {int(vf)}')


def run_randomwalk(name, num_RWs=20, num_seeds=100):

    print(f'running: random walk {name}')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/nonNN/{name}'

    val_matrix = np.zeros((num_RWs,
                           num_seeds))
    print(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}')

    # pack inputs for multiprocessing map
    mp_inputs = []
    for n in range(num_RWs):
        for s in range(num_seeds):
            mp_inputs.append( (None, None, None, s, None) ) # model_tuple=None, pv=None, load_dir=None, env_path=None

    # run agents in parallel
    with mp.Pool() as pool:
        results = pool.starmap_async(start, mp_inputs)
        pool.close()
        pool.join()
    results_list = results.get()

    # skip to start of each seed series/chunk + allocate fitness to save matrix
    for i,c in enumerate(range(0, len(results_list), num_seeds)):
        for s,fitnesses in enumerate(results_list[c : c + num_seeds]):
            val_matrix[i,s] = round(fitnesses[0],0)

    # saving protocol 

    with open(fr'{exp_path}.bin', 'wb') as f:
        pickle.dump(val_matrix, f)
    avg_per_NN = np.average(val_matrix).round(1)

    print(f'val_fit: {int(avg_per_NN)}')


def run_perfect(name='perfect', num_RWs=1, num_seeds=100):

    print(f'running: perfect')

    data_dir = Path(__file__).parent.parent / r'data/simulation_data/'
    exp_path = fr'{data_dir}/nonNN/{name}'

    val_matrix = np.zeros((num_RWs,
                           num_seeds))
    print(f'Validation matrix shape (num_NNs, num_seeds): {val_matrix.shape}')

    x_min, x_max = 20,980
    y_min, y_max = 20,980
    target = np.array([400,400])
    res_radius = 50
    agent_max_vel = 2

    # numerically compute perfect trajectories
    for s in range(num_seeds):
        
        # fire two blanks before location call
        _ = np.random.randint(x_min, x_max)
        _ = np.random.randint(x_min, x_max)

        x = np.random.randint(x_min, x_max)
        y = np.random.randint(y_min, y_max)
        # orient = np.random.uniform(0, 2*np.pi)

        start = np.array([x,y])

        dist = np.linalg.norm(start - target)
        dist_to_edge = dist - res_radius

        time_to_edge = dist_to_edge/agent_max_vel

        val_matrix[0,s] = time_to_edge

    # saving protocol 
    with open(fr'{exp_path}.bin', 'wb') as f:
        pickle.dump(val_matrix, f)

    print(f'val_fit: {int(np.average(val_matrix))}')



if __name__ == '__main__':

    names = []

    # for name in names:
    #     rerun_NNs(name)


    # run_randomwalk('rotdiff_0p001')
    # run_randomwalk('rotdiff_0p005')
    # run_randomwalk('rotdiff_0p01')
    # run_randomwalk('rotdiff_0p05')
    # run_randomwalk('rotdiff_0p10')
    # run_randomwalk('rotdiff_0p50')
    # run_perfect()

    # # vis
    # for name in [f'sc_CNN14_FNN2_p50e20_vis6_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis10_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis14_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis20_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)

    # # cnn
    # for name in [f'sc_CNN12_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN13_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN15_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN16_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)

    # # fnn
    # for name in [f'sc_CNN14_FNN1_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # # for name in [f'sc_CNN14_FNN3_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # for name in [f'sc_CNN14_FNN4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN8_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2x2_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2x3_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # for name in [f'sc_CNN14_FNN2x4_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2x8_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # for name in [f'sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    
    # # fov
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov2_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov3_rep{x}' for x in range(20)]:
    #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov35_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov45_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov5_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov6_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov7_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_fov875_rep{x}' for x in range(20)]:
    #     names.append(name)

    # # dist
    # # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_minmax_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_rep{x}' for x in range(20)]:
    # #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for s in [10000,20000,30000,40000]:
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    # for s in [10000,20000,30000,40000]:
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis12_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    # for s in [10000,20000,30000,40000]:
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    # for s in [10000,20000,30000,40000]:
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis24_PGPE_ss20_mom8_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    # for s in [10000,20000,30000,40000]:
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)
    # for s in [10000,20000,30000,40000]:
    #     for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_sWF_n0_seed{str(int(s/1000))}k_rep{x}' for x in range(20)]:
    #         names.append(name)

    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n1_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n2_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n3_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_WF_n4_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis8_lm100_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis10_lm100_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis12_lm100_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis16_lm100_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis24_lm100_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_lm_CNN14_FNN2_p50e20_vis32_lm100_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_rep{x}' for x in range(20)]:
    #    names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacehalf_seed10k_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_actspacenarrow_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2x16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_GRUpara16_p50e20_vis8_PGPE_ss20_mom8_rep{x}' for x in range(20)]:
    #     names.append(name)


    # # boundary_scale

    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound500_rep{x}' for x in range(20)]:
    #    names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound500_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_maxWF_n0_bound500_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_PGPE_ss20_mom8_dist_mlWF_n0_bound500_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN24_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN17_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN16_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN27_FNN16_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)

    # for name in [f'sc_CNN24_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN17_FNN2_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN16_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN27_FNN16_p50e20_vis8_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN1148_FNN2_p50e20_vis32_PGPE_ss20_mom8_bound1000_rep{x}' for x in range(20)]:
    #     names.append(name)


    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis32_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
        
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_maxWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p9WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_p8WF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mlWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_mWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_msWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_sWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_PGPE_ss20_mom8_dist_ssWF_n0_rep{x}' for x in range(20)]:
    #     names.append(name)



    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_2xpinball_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis16_2xpinball_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN14_FNN2_p50e20_vis8_maxWF_2xpinball_rep{x}' for x in range(20)]:
    #     names.append(name)
    # for name in [f'sc_CNN17_FNN16_p50e20_vis16_2xpinball_rep{x}' for x in range(20)]:
    #     names.append(name)

    n = 5
    for name in [f'nowall_N5_CNN14_FNN2_vis8_rep{x}' for x in range(n)]:
        names.append(name)
    for name in [f'nowall_N5_CNN14_FNN2_vis16_rep{x}' for x in range(n)]:
        names.append(name)
    for name in [f'nowall_N5_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
        names.append(name)
    # for name in [f'nowall_N10_CNN14_FNN2_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'nowall_N10_CNN14_FNN2_vis16_rep{x}' for x in range(n)]:
    #     names.append(name)
    # for name in [f'nowall_N10_CNN14_FNN16_vis8_rep{x}' for x in range(n)]:
    #     names.append(name)

    for name in names:
        rerun_NNs(name)
        # rerun_NNs(name, noise_type='angle_n10')
        # rerun_NNs(name, noise_type='dist_n025')
