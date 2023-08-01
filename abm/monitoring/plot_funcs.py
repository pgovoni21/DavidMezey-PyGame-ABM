import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle

def plot_map(plot_data, x_max, y_max, w=4, h=4, save_name=None):

    ag_data, res_data = plot_data

    fig, axes = plt.subplots() 
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)

    # rescale plotting area to square
    l,r,t,b = fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom
    fig.set_size_inches( float(w)/(r-l) , float(h)/(t-b) )

    # resource patches via circles
    N_res = res_data.shape[0]
    for res in range(N_res):
        
        # unpack data array
        pos_x = res_data[res,0,0]
        pos_y = res_data[res,0,1]
        radius = res_data[res,0,2]

        axes.add_patch( plt.Circle((pos_x,pos_y), radius, color='lightgray', zorder=0) )
    
    # agent trajectories as arrows/points/events
    N_ag = ag_data.shape[0]
    for agent in range(N_ag):

        # unpack data array
        pos_x = ag_data[agent,:,0]
        pos_y = ag_data[agent,:,1]
        mode_nums = ag_data[agent,:,2]

        # agent start/end points
        axes.plot(pos_x[0], pos_y[0], 'wo', ms=10, markeredgecolor='k', zorder=4, clip_on=False)
        axes.plot(pos_x[-1], pos_y[-1], 'ko', ms=10, zorder=4, clip_on=False)

        # build arrays according to agent mode (~5x faster plot runtime than using compressed save_data)
        traj_explore, traj_exploit, traj_collide = [],[],[]
        for x, y, mode_num in zip(pos_x, pos_y, mode_nums):
            if mode_num == 0: traj_explore.append([x,y])
            elif mode_num == 1: traj_exploit.append([x,y])
            elif mode_num == 2: traj_collide.append([x,y])
        traj_explore, traj_exploit, traj_collide = np.array(traj_explore), np.array(traj_exploit), np.array(traj_collide)
        
        # add agent directional trajectory via arrows 
        arrows(axes, traj_explore[:,0], traj_explore[:,1])

        # add agent positional trajectory + mode via points
        axes.plot(traj_explore[:,0], traj_explore[:,1],'o', color='royalblue', ms=.5, zorder=2)
        if traj_exploit.size: axes.plot(traj_exploit[:,0], traj_exploit[:,1],'o', color='green', ms=5, zorder=3)
        if traj_collide.size: axes.plot(traj_collide[:,0], traj_collide[:,1],'o', color='red', ms=5, zorder=3, clip_on=False)

        ### SLOW ###
        # # agent directional trajectory via arrows 
        # self.arrows(axes, pos_x, pos_y)
        # # agent positional trajectory + mode via points
        # axes.plot(pos_x, pos_y,'o', color='royalblue', ms=.5, zorder=2) # exploring, small blue --> taken out of for-loop for speed
        # for x, y, mode_num in zip(pos_x, pos_y, mode_nums):
        #     if mode_num == 2: axes.plot(x, y,'o', color='green', ms=5, zorder=3) # exploiting, large green
        #     elif mode_num == 3: axes.plot(x, y,'o', color='red', ms=5, zorder=3) # colliding, large red

    if save_name:
        # line added to sidestep backend memory issues in matplotlib 3.5+
        # if not used, (sometimes) results in tk.call Runtime Error: main thread is not in main loop
        # though this line results in blank first plot, not sure what to do here
        matplotlib.use('Agg')
        
        plt.savefig(fr'{save_name}.png')
        plt.close()
    else:
        plt.show()

    # for live plotting during an evol run - pull details to EA class for continuous live plotting
    # implement this --> https://stackoverflow.com/a/49594258
    # plt.clf
    # plt.draw()
    # plt.pause(0.001)

def arrows(axes, x, y, ahl=6, ahw=3):
    # from here: https://stackoverflow.com/questions/8247973/how-do-i-specify-an-arrow-like-linestyle-in-matplotlib

    # r is the distance spanned between pairs of points
    r = [0]
    for i in range(1,len(x)):
        dx = x[i]-x[i-1]
        dy = y[i]-y[i-1]
        r.append(np.sqrt(dx*dx+dy*dy))
    r = np.array(r)

    # set arrow spacing
    num_arrows = int(len(x) / 15)
    aspace = r.sum() / (num_arrows + 1)
    
    # set inital arrow position at first space
    arrowPos = 0
    
    # rtot is a cumulative sum of r, it's used to save time
    rtot = []
    for i in range(len(r)):
        rtot.append(r[0:i].sum())
    rtot.append(r.sum())

    arrowData = [] # will hold tuples of x,y,theta for each arrow

    ndrawn = 0
    rcount = 1 
    while arrowPos < r.sum() and ndrawn < num_arrows:
        x1, x2 = x[rcount-1], x[rcount]
        y1, y2 = y[rcount-1], y[rcount]
        da = arrowPos - rtot[rcount]
        theta = np.arctan2((x2-x1),(y2-y1))
        ax = np.sin(theta)*da + x1
        ay = np.cos(theta)*da + y1
        arrowData.append((ax,ay,theta))
        ndrawn += 1
        arrowPos += aspace
        while arrowPos > rtot[rcount+1]: 
            rcount += 1
            if arrowPos > rtot[-1]:
                break

    for ax,ay,theta in arrowData:
        # use aspace as a guide for size and length of things
        # scaling factors were chosen by experimenting a bit

        dx0 = np.sin(theta)*ahl/2. + ax
        dy0 = np.cos(theta)*ahl/2. + ay
        dx1 = -1.*np.sin(theta)*ahl/2. + ax
        dy1 = -1.*np.cos(theta)*ahl/2. + ay

        axes.annotate('', xy=(dx0, dy0), xytext=(dx1, dy1),
                arrowprops=dict( headwidth=ahw, headlength=ahl, ec='royalblue', fc='royalblue', zorder=1))
        

def plot_EA_trend_violin(trend_data, save_dir=False):

    # convert to array
    trend_data = np.array(trend_data)

    # transpose from shape: (number of generations, population size)
    #             to shape: (population size, number of generations)
    data_per_gen_tp = trend_data.transpose()

    # plot population distributions + means of fitnesses for each generation
    plt.violinplot(data_per_gen_tp, widths=1, showmeans=True, showextrema=False)

    if save_dir: 
        plt.savefig(fr'{save_dir}/fitness_spread_violin.png')
    else: 
        plt.show()

    plt.close()




def plot_mult_EA_trends(names):

    # establish load directory
    root_dir = Path(__file__).parent.parent
    data_dir = Path(root_dir, r'data/simulation_data')

    # init plot details
    fig, ax1 = plt.subplots(figsize=(15,10)) 
    # ax2 = ax1.twinx()
    cmap = plt.get_cmap('hsv')
    cmap_range = len(names)
    lns = []
    
    # iterate over each file
    for i, name in enumerate(names):

        with open(fr'{data_dir}/{name}/fitness_spread_per_generation.bin','rb') as f:
            trend_data = pickle.load(f)


        # data_per_gen_tp = np.array(trend_data).transpose()
        # plt.violinplot(data_per_gen_tp, widths=1, showmeans=True, showextrema=False)
        # plt.savefig(fr'{data_dir}/{name}/fitness_spread_violin.png')
        # plt.close()

        run_data_exists = False
        if Path(fr'{data_dir}/{name}/run_data.bin').is_file():
            run_data_exists = True
            with open(fr'{data_dir}/{name}/run_data.bin','rb') as f:
                mean_pv, std_pv, time = pickle.load(f)
            print(f'{name}, time taken: {time}')

            # x=plt.imshow(mean_pv.transpose())
            # x=plt.imshow(std_pv.transpose())
            # plt.colorbar(x)
            # plt.violinplot(std_pv.transpose(), widths=1, showmeans=True, showextrema=False)
            # # plt.violinplot(mean_pv.transpose(), widths=1, showmeans=True, showextrema=False)
            # plt.xlabel('Generation')
            # plt.ylabel('Parameter')

            # plt.savefig(fr'{data_dir}/{name}/std_spread_violin.png')
            # plt.savefig(fr'{data_dir}/{name}/mean_spread_violin.png')
            # plt.show()
            # plt.close()


        trend_data = np.array(trend_data)

        best_trend_data = np.min(trend_data, axis=1)
        l1 = ax1.plot(best_trend_data, 
                        label = f'max {name}',
                        # label = f'avg {name} | t: {time} sec',
                        color=cmap(i/cmap_range), 
                        linestyle='dotted'
                        )
        lns.append(l1[0])

        avg_trend_data = np.mean(trend_data, axis=1)

        if run_data_exists:
            l2 = ax1.plot(avg_trend_data, 
                            # label = f'avg {name}',
                            label = f'avg {name} | t: {int(time)} sec',
                            color=cmap(i/cmap_range), 
                            # linestyle='dashed'
                            )
        else:
            l2 = ax1.plot(avg_trend_data, 
                            label = f'avg {name}',
                            # label = f'avg {name} | t: {int(time)} sec',
                            color=cmap(i/cmap_range), 
                            # linestyle='dashed'
                            )
        lns.append(l2[0])
    
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Time to Find Patch')

    # for corner patch search
    ax1.set_ylim(-20,1020)

    # plt.savefig(fr'{data_dir}/stationarycorner_FNN_width.png')
    # plt.savefig(fr'{data_dir}/stationarycorner_CNN_width.png')
    # plt.savefig(fr'{data_dir}/stationarycorner_ep_num.png')
    # plt.savefig(fr'{data_dir}/stationarycorner_pop_size.png')
    # plt.savefig(fr'{data_dir}/stationarycorner_init_sig.png')
    # plt.savefig(fr'{data_dir}/stationarycorner_FNN_width_fast.png')
    # plt.savefig(fr'{data_dir}/stationarycorner_other_input.png')

    plt.savefig(fr'{data_dir}/stationarypoint.png')

    plt.show()


if __name__ == '__main__':

    names = []

    # CNN = 14
    # FNN = 8
    # p = 50
    # e = 10
    # sig = '0p1'

    # FNN_iter = [
    #     4,5,6,7,8,16
    # ]
    # CNN_iter = [
    #     12,13,14,18
    # ]
    # e_iter = [
    #     5,10,15
    # ]
    # p_iter = [
    #     25,50,100
    # ]
    # sig_iter = [
    #     '0p1',1,10
    # ]

    # CNN = 12
    # p = 25
    # e = 5
    # sig = '0p1'
    # FNN_iter = [
    #     1,2,4,6
    # ]

    # for FNN in FNN_iter:
    # # for CNN in CNN_iter:
    # # for e in e_iter:
    # # for p in p_iter:
    # # for sig in sig_iter:
    #     name = f'stationarycorner_CNN{str(CNN)}_FNN{str(FNN)}_p{str(p)}e{str(e)}_sig{str(sig)}'
    #     names.append(name)

    # names.append('stationarycorner_CNN14_FNN8_p25e10_sig0p1')
    # names.append('stationarycorner_CNN14_FNN8_p50e5_sig0p1')
    # names.append('stationarycorner_CNN12_FNN8_p50e10_sig0p1')
    
    # names.append('stationarycorner_CNN12_FNN1_p25e5_sig0p1')
    # names.append('stationarycorner_CNN12_FNN1_p25e5_sig0p1_other0')

    # # rnn_hidden_iter = [8,16]
    # rnn_hidden_iter = [8]
    # # cnn_dims_iter = [4,8]
    # cnn_dims_iter = [4]
    # cnn_depths_iter = [1,2]
    # # rnn_type_iter = ['fnn','ctrnn','gru']
    # rnn_type_iter = ['fnn']

    # for l in rnn_type_iter:
    #     for i in rnn_hidden_iter:
    #         for j in cnn_dims_iter:
    #             for k in cnn_depths_iter:

    #                 name = f'stationarypoint_CNN{str(k)}{str(j)}_{l.upper()}{str(i)}_p50e10g500_sig0p1'
    #                 names.append(name)

    # names.append('stationarypoint_CNN18_FNN8_p50e10g500_sig0p1')
    # names.append('stationarypoint_CNN18_FNN8_p25e5_sig0p1')
    # names.append('stationarypoint_CNN18_CTRNN8_p25e5_sig0p1')

    names.append('stationarypoint_CNN14_FNN8_p50e10g500_sig0p1')
    names.append('stationarypoint_CNN18_FNN8_p50e10g500_sig0p1')
    names.append('stationarypoint_CNN24_FNN8_p50e10g500_sig0p1')


    names.append('stationarypoint_CNN104_FNN8_p50e10g500_sig0p1')
    names.append('stationarypoint_CNN108_FNN8_p50e10g500_sig0p1')
    names.append('stationarypoint_CNN108_FNN8_p25e5g500_sig0p1')


    
    # names.append('stationarypoint_CNN18_FNN8_p25e5g500_sig10')
    names.append('stationarypoint_CNN18_FNN8_p25e5g500_sig0p1_vis16')
    names.append('stationarypoint_CNN1148_FNN8_p25e5g500_sig0p1')
    # names.append('stationarypoint_CNN1148_FNN8_p25e5g500_sig10')
    


    # names.append('stationarypoint_CNN18_FNN8_p25e5_sig0p1')
    # names.append('stationarypoint_CNN18_FNN8_p25e5g500_sig10')

    plot_mult_EA_trends(names)