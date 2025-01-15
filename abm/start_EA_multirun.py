from abm.start_EA import start_EA

from pathlib import Path
import dotenv as de


def set_env_var(key, val):
    env_path = Path(__file__).parent.parent / ".env"
    de.set_key(env_path, str(key), str(val))


def EA_runner():

# compute4
    # for x in range(5):
    #     set_env_var('N', '5')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', '8')
    #     set_env_var('RNN_HIDDEN_SIZE', '2')
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('EA_EPISODES', '40')
    #     set_env_var('SIM_TYPE', 'nowalls')
    #     set_env_var('EA_SAVE_NAME', f'nowall_N5_CNN14_FNN2_vis8_e40_rep{x}')
    #     start_EA()

# compute1
    # for x in range(5):
    #     set_env_var('N', '5')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', '16')
    #     set_env_var('RNN_HIDDEN_SIZE', '2')
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('EA_EPISODES', '40')
    #     set_env_var('SIM_TYPE', 'nowalls')
    #     set_env_var('EA_SAVE_NAME', f'nowall_N5_CNN14_FNN2_vis16_e40_rep{x}')
    #     start_EA()

# compute7
    for x in range(4):
        set_env_var('N', '5')
        set_env_var('VISUAL_FIELD_RESOLUTION', '8')
        set_env_var('RNN_HIDDEN_SIZE', '16')
        set_env_var('RNN_TYPE', 'fnn')
        set_env_var('EA_EPISODES', '40')
        set_env_var('SIM_TYPE', 'nowalls')
        set_env_var('EA_SAVE_NAME', f'nowall_N5_CNN14_FNN16_vis8_e40_rep{x+1}')
        start_EA()

# compute2
    # for x in range(3):
    #     set_env_var('N', '10')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', '8')
    #     set_env_var('RNN_HIDDEN_SIZE', '2')
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('EA_EPISODES', '20')
    #     set_env_var('SIM_TYPE', 'nowalls')
    #     set_env_var('EA_SAVE_NAME', f'nowall_N10_CNN14_FNN2_vis8_rep{x+2}')
    #     start_EA()
    # set_env_var('N', '10')
    # set_env_var('VISUAL_FIELD_RESOLUTION', '8')
    # set_env_var('RNN_HIDDEN_SIZE', '16')
    # set_env_var('RNN_TYPE', 'fnn')
    # set_env_var('EA_EPISODES', '20')
    # set_env_var('SIM_TYPE', 'nowalls')
    # set_env_var('EA_SAVE_NAME', f'nowall_N10_CNN14_FNN16_vis8_rep3')
    # start_EA()

# compute3
    # for x in range(3):
    #     set_env_var('N', '10')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', '16')
    #     set_env_var('RNN_HIDDEN_SIZE', '2')
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('EA_EPISODES', '20')
    #     set_env_var('SIM_TYPE', 'nowalls')
    #     set_env_var('EA_SAVE_NAME', f'nowall_N10_CNN14_FNN2_vis16_rep{x+2}')
    #     start_EA()
    # set_env_var('N', '10')
    # set_env_var('VISUAL_FIELD_RESOLUTION', '8')
    # set_env_var('RNN_HIDDEN_SIZE', '16')
    # set_env_var('RNN_TYPE', 'fnn')
    # set_env_var('EA_EPISODES', '20')
    # set_env_var('SIM_TYPE', 'nowalls')
    # set_env_var('EA_SAVE_NAME', f'nowall_N10_CNN14_FNN16_vis8_rep4')
    # start_EA()

# compute8
    # for x in range(5):
    #     set_env_var('N', '10')
    #     set_env_var('VISUAL_FIELD_RESOLUTION', '8')
    #     set_env_var('RNN_HIDDEN_SIZE', '16')
    #     set_env_var('RNN_TYPE', 'fnn')
    #     set_env_var('EA_EPISODES', '20')
    #     set_env_var('SIM_TYPE', 'nowalls_ghostexploiter')
    #     set_env_var('EA_SAVE_NAME', f'nowall_N10_CNN14_FNN16_vis8_e20_ghost_rep{x}')
    #     start_EA()


if __name__ == '__main__':

    EA_runner()