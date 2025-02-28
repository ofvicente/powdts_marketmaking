import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle

from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.market_makers.AdaptiveMarketMakerAgent import AdaptiveMarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent
from model.LatencyModel import LatencyModel

from agent.paper_MMInvestorAgent import MMInvestorAgent
from agent.market_makers.MORL_MM_Agent import DQL_MarketMakerQL

import os
import json
import pickle
import gc

import warnings
warnings.filterwarnings('ignore')
os.environ["KMP_WARNINGS"] = "FALSE"


def main():
    ########################################################################################################################
    ############################################### GENERAL CONFIG #########################################################
    ########################################################################################################################

    parser = argparse.ArgumentParser(description='Detailed options for config O.')

    parser.add_argument('-c',
                        '--config',
                        required=True,
                        help='Name of config file to execute')
    parser.add_argument('-t',
                        '--ticker',
                        required=True,
                        help='Ticker (symbol) to use for simulation')
    parser.add_argument('-d', '--historical-date',
                        required=True,
                        type=parse,
                        help='historical date being simulated in format YYYYMMDD.')
    parser.add_argument('--start_time',
                        default='09:30:00',
                        type=parse,
                        help='Starting time of simulation.'
                        )
    parser.add_argument('--end_time',
                        default='11:30:00',
                        type=parse,
                        help='Ending time of simulation.'
                        )
    parser.add_argument('-l',
                        '--log_dir',
                        default=None,
                        help='Log directory name (default: unix timestamp at program start)')
    parser.add_argument('-s',
                        '--seed',
                        type=int,
                        default=None,
                        help='numpy.random.seed() for simulation')
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='Maximum verbosity!')
    parser.add_argument('--config_help',
                        action='store_true',
                        help='Print argument options for this config file')
    # Execution agent config
    parser.add_argument('-e',
                        '--execution-agents',
                        action='store_true',
                        help='Flag to allow the execution agent to trade.')
    parser.add_argument('-p',
                        '--execution-pov',
                        type=float,
                        default=0.1,
                        help='Participation of Volume level for execution agent')
    # market maker config
    parser.add_argument('--mm-pov',
                        type=float,
                        default=0.025
                        )
    parser.add_argument('--mm-window-size',
                        type=util.validate_window_size,
                        default='adaptive'
                        )
    parser.add_argument('--mm-min-order-size',
                        type=int,
                        default=1
                        )
    parser.add_argument('--mm-num-ticks',
                        type=int,
                        default=10
                        )
    parser.add_argument('--mm-wake-up-freq',
                        type=str,
                        default='10S'
                        )
    parser.add_argument('--mm-skew-beta',
                        type=float,
                        default=0
                        )
    parser.add_argument('--mm-level-spacing',
                        type=float,
                        default=5
                        )
    parser.add_argument('--mm-spread-alpha',
                        type=float,
                        default=0.75
                        )
    parser.add_argument('--mm-backstop-quantity',
                        type=float,
                        default=50000)

    parser.add_argument('--fund-vol',
                        type=float,
                        default=1e-8,
                        help='Volatility of fundamental time series.'
                        )

    parser.add_argument('--exp',
                        required=True,
                        help='Experiment folder')

    parser.add_argument('--round_exp',
                        required=False,
                        help='Parallel round of experiment',
                        default=None)

    parser.add_argument('--onlyrandom',
                        required=False,
                        help='Only random agents?',
                        default=False)

    parser.add_argument('--qtable_file',
                        required=False,
                        help='File with pretrained qtable',
                        default=False)

    parser.add_argument('--dql_file_1',
                        required=False,
                        help='File with pretrained neural network 1',
                        default=False)

    parser.add_argument('--dql_file_2',
                        required=False,
                        help='File with pretrained neural network 2',
                        default=False)
    
    parser.add_argument('--dql_file_3',
                        required=False,
                        help='File with pretrained neural network 3',
                        default=False)

    parser.add_argument('--dql_set_files',
                        required=False,
                        help='Set File with pretrained neural network',
                        default=False)

    parser.add_argument('--dql_comp_file_1',
                        required=False,
                        help='Set COMPETITOR File with pretrained neural network',
                        default=False)

    parser.add_argument('--rewardPenalty',
                        required=False,
                        help='Method for penalty of rewards calculation',
                        default=False)

    parser.add_argument('--penaltyMultiplier',
                        required=False,
                        help='Penalty of rewards calculation',
                        default=0)                    

    parser.add_argument('--learningRate',
                        required=False,
                        help='Neural Network Learning Rate',
                        default=0.01)

    parser.add_argument('--neuralModel',
                        required=False,
                        help='Neural Network model',
                        default=0)

    parser.add_argument('--clone',
                        required=False,
                        help='Cloning',
                        default=False)

    parser.add_argument('--adversarial',
                        required=False,
                        help='Adversarial',
                        default=False)
    
    parser.add_argument('--additionalql',
                        required=False,
                        help='Additional QL',
                        default=False)

    parser.add_argument('--clonning_turbo_epsilon',
                        required=False,
                        help='Cloning turbo epsilons',
                        default=False)

    parser.add_argument('--main_turbo_epsilon',
                        required=False,
                        help='Main turbo epsilons',
                        default=False)

    parser.add_argument('--turbo_epsilon_repeats',
                        required=False,
                        help='Main turbo epsilons repetition',
                        default=1)

    parser.add_argument('--reduced_states',
                        required=False,
                        help='Reduced state space',
                        default=False)

    parser.add_argument('--qlagents',
                        required=False,
                        help='Number of QL agents',
                        default=1)

    parser.add_argument('--nsims',
                        required=False,
                        help='Number of simulations',
                        default=150)

    parser.add_argument('--numrandom',
                        required=False,
                        help='Number of random agents',
                        default=1)

    parser.add_argument('--numpersistent',
                        required=False,
                        help='Number of persistent agents',
                        default=1)

    parser.add_argument('--avoid_retrain',
                        required=False,
                        help='Avoid retraining of neural network (fits). Test mode',
                        default=False)

    parser.add_argument('--avoid_retrain_ql2',
                        required=False,
                        help='Avoid retraining of neural network (fits). Test mode ON QL2',
                        default=False)

    parser.add_argument('--numinvestors',
                        required=False,
                        help='Number of investor agents',
                        default=50)

    parser.add_argument('--keep_inv_cash',
                        required=False,
                        help='Keep inventory and cash',
                        default=False)

    parser.add_argument('--alphareward',
                        required=False,
                        help='Penalty reward alpha',
                        default=False)

    parser.add_argument('--hedges',
                        required=False,
                        help='Include hedging',
                        default=True)

    parser.add_argument('--multi_weight',
                        required=True,
                        help='First percentage of multi objective = e.g. 0.3 -> (0.3*Obj1 + 0.7*obj2)')

    parser.add_argument('--finetuning_epsilon',
                        required=False,
                        help='Finetuning epsilon',
                        default=0.5)

    parser.add_argument('--finetuning_rbuff_mix',
                        required=False,
                        help='Finetuning rbuff mix',
                        default=0.5)
    
    parser.add_argument('--apply_old_replay_buffer',
                        required=False,
                        help='Apply old replay buffer',
                        default=False)
    
    parser.add_argument('--th_alpha',
                        required=True,
                        help='Thompson sampling alpha parameter',
                        default=False)
    
    parser.add_argument('--th_beta',
                        required=True,
                        help='Thompson sampling beta parameter',
                        default=False)
    
    parser.add_argument('--th_gamma',
                        required=True,
                        help='Thompson sampling gamma parameter',
                        default=0.8)

    args, _ = parser.parse_known_args()

    if args.config_help:
        parser.print_help()
        sys.exit()

    seed = args.seed
    if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
    np.random.seed(seed)
    randomstate = np.random.RandomState(seed=seed)

    util.silent_mode = not args.verbose
    LimitOrder.silent_mode = not args.verbose

    exchange_log_orders = False
    log_orders = False
    book_freq = None

    ##########################
    # EXP Hyperparameters
    ##########################

    n_states = 100
    n_actions = 121
    force_same_epsilon = False
    alpha = 0.99
    alpha_decay = 1
    epsilon = 0.99
    epsilon_decay = 0.99993
    epsilon_min = 0.01
    gamma = 0.98

    qtable_file = args.qtable_file
    dql_file_1 = args.dql_file_1
    dql_file_2 = args.dql_file_2
    dql_file_3 = args.dql_file_3
    dql_comp_file_1 = args.dql_comp_file_1

    finetuning_rbuff_mix = float(args.finetuning_rbuff_mix)
    finetuning_epsilon = float(args.finetuning_epsilon)

    loaded_nn_1 = False
    loaded_nn_2 = False
    loaded_nn_set = False
    use_optimal_dql = False

    no_inventory = False
    include_hedges = bool(int(args.hedges))

    save_results = True

    num_simulations = int(args.nsims)

    experiment_folder = "./exper_results/"+str(args.exp)+"/"
    round_exp = str(args.round_exp)

    if(no_inventory==True or include_hedges==False):
        print("\nATTENTION!!!!!!! \nINVENTORY REWARD NOT ACTIVE (False - True)!!!!!!!!!\n\n", no_inventory, include_hedges)

    if(args.onlyrandom):
        epsilon = 1
        epsilon_decay = 1
        print("FORCING RANDOM!!!")

    if(args.qtable_file):
        print("USING PREVIOUS QTABLE:", qtable_file)
        loaded_qtable = pickle.load(open(qtable_file, 'rb'))
        loaded_qtable.epsilon = 0
        loaded_qtable.epsilon_min = 0

    if(args.dql_file_1):
        if(args.dql_file_1.lower() == "optimal"):
            print("*:*:*:*:*:*: USING OPTIMAL from dict! DQL Neural Network PREVIOUS:", dql_file_1)
            use_optimal_dql = True
        else:                
            print("*:*:*:*:*:*: USING DQL Neural Network PREVIOUS:", dql_file_1)
            loaded_nn_1 = pickle.load(open(dql_file_1, 'rb'))
        epsilon = 0
        epsilon_min = 0

    if(args.dql_file_2):
        print("+.+.+.+.+.+.+.+.+ USING 2nd SECOND DQL Neural Network PREVIOUS:", dql_file_2)
        loaded_nn_2 = pickle.load(open(dql_file_2, 'rb'))
        epsilon = 0
        epsilon_min = 0

    if(args.dql_file_3):
        print("+.+.+.+.+.+.+.+.+ USING 3rd THIRD DQL Neural Network PREVIOUS:", dql_file_3)
        loaded_nn_3 = pickle.load(open(dql_file_3, 'rb'))
        epsilon = 0
        epsilon_min = 0

    if(args.dql_comp_file_1):
        print("+.+.+.+.+.+.+.+.+ USING COMPETITOR DQL Neural Network PREVIOUS:", dql_comp_file_1)

    if(args.rewardPenalty):
        rewardPenalty = int(args.rewardPenalty)
    else:
        rewardPenalty = 0

    if(args.penaltyMultiplier):
        penaltyMultiplier = float(args.penaltyMultiplier)
    else:
        penaltyMultiplier = 0

    print("+ USING REWARD PENALTY:", rewardPenalty)
    print("+ USING finetuning_epsilon:", finetuning_epsilon)

    if(args.learningRate):
        learningRate = float(args.learningRate)
    else:
        learningRate = 0.01

    if(args.neuralModel):
        neuralModel = int(args.neuralModel)
    else:
        neuralModel = 0

    if(args.clone):
        num_behaviouralclonners = True
        print("**** CLONING *****", num_behaviouralclonners)
    else:
        num_behaviouralclonners = False

    if(args.adversarial):
        num_adversarial = True
        print("**** ADVERSARIAL *****", num_adversarial)
    else:
        num_adversarial = False

    if(args.additionalql):
        num_additional_forced = int(args.additionalql)
        print("**** FORCING ADDITIONAL # OF QLs *****", num_additional_forced)
    else:
        num_additional_forced = False

    if(args.clonning_turbo_epsilon):
        turbo_epsilon = 2
        print("**** TURBO EPSILON FOR CLONING ACTIVATED *****", turbo_epsilon)
    else:
        turbo_epsilon = 1

    if(args.main_turbo_epsilon):
        main_turbo_epsilon = 2
        print("**** TURBO EPSILON MAIN ACTIVATED *****", main_turbo_epsilon)
    else:
        main_turbo_epsilon = 1

    if(args.turbo_epsilon_repeats):
        turbo_epsilon_repeats = int(args.turbo_epsilon_repeats)
        print("**** TURBO EPSILON MAIN CYCLE REPEATS ACTIVATED *****", turbo_epsilon_repeats)
    else:
        turbo_epsilon_repeats = 1

    if(args.reduced_states):
        reduced_states = True
        print("**** STATE SPACE REDUCED! *****", reduced_states)
    else:
        reduced_states = False

    if(args.avoid_retrain):
        avoid_retrain = bool(args.avoid_retrain)
        print("**** AVOID RETRAIN! *****", avoid_retrain)
    else:
        avoid_retrain = False

    if(args.avoid_retrain_ql2):
        avoid_retrain_ql2 = bool(args.avoid_retrain_ql2)
        print("**** AVOID RETRAIN QL 2 ! *****", avoid_retrain_ql2)
    else:
        avoid_retrain_ql2 = False

    if(args.th_alpha):
        th_alpha = int(args.th_alpha)
        print("**** TH ALPHA: *****", th_alpha)
    else:
        th_alpha = 1

    if(args.th_beta):
        th_beta = int(args.th_beta)
        print("**** TH BETA: *****", th_beta)
    else:
        th_beta = 1

    if(args.th_gamma):
        th_gamma = float(args.th_gamma)
        print("**** TH GAMMA: *****", th_gamma)
    else:
        th_gamma = 0.85

    if(args.keep_inv_cash):
        print("**** $$$$ Keeping inventory and cash for everyone")

    loaded_set_json = False    
    old_replay_buffer_file = False

    multi_weight = args.multi_weight

    n_e = 0
    main_mm_dqn_id = None
    main_dqn_last_state = None
    main_dqn_list = []

    # the different markets to be simulated in ABIDES
    exp_loop_conf = [{"noise":100, "val":10, "mom":10, "mm":1, "mmticks":10, "additional_ql":0},                       
                    {"noise":100, "val":10, "mom":10, "mm":1, "mmticks":10, "additional_ql":5},
                    {"noise":100, "val":10, "mom":10, "mm":1, "mmticks":10, "additional_ql":1},
                    {"noise":100, "val":10, "mom":10, "mm":1, "mmticks":10, "additional_ql":7},
                    {"noise":100, "val":10, "mom":10, "mm":1, "mmticks":10, "additional_ql":1},
                    {"noise":100, "val":10, "mom":10, "mm":1, "mmticks":10, "additional_ql":7},
                    {"noise":100, "val":10, "mom":10, "mm":1, "mmticks":10, "additional_ql":5},
                    {"noise":100, "val":10, "mom":10, "mm":1, "mmticks":10, "additional_ql":0}]


    # Optimal DQLs configuration to be loaded - Policies
    
    optimal_dqls = {0: '/obj/pretrained_policies/policy0.pkl',
                    1: '/obj/pretrained_policies/policy1.pkl',
                    5: '/obj/pretrained_policies/policy5.pkl',
                    7: '/obj/pretrained_policies/policy7.pkl'}

    for e in exp_loop_conf:
        # Preload optimal neural network for the QL
        if(use_optimal_dql):           
            print("*:*:*:*:*:*: LOADING OPTIMAL DQL Neural Network PREVIOUS:", e["additional_ql"], optimal_dqls[e["additional_ql"]])
            loaded_nn_1 = pickle.load(open(optimal_dqls[e["additional_ql"]], 'rb'))

        num_noise = e["noise"]
        num_value = e["val"]
        num_momentum_agents = e["mom"]
        num_mm_agents = e["mm"]
        additional_ql = e["additional_ql"]
        mm_num_ticks = e["mmticks"]

        mm_params = [(args.mm_window_size, args.mm_pov, mm_num_ticks, args.mm_wake_up_freq, args.mm_min_order_size),
                    (args.mm_window_size, args.mm_pov, mm_num_ticks, args.mm_wake_up_freq, args.mm_min_order_size)]
        
        num_marketMakersAgents = int(args.numinvestors)
        num_qlearnersMM = int(args.qlagents)
        num_behaviouralclonners = False
        num_adversarial = False
        num_randomMM = int(args.numrandom)
        num_persistentMM = int(args.numpersistent)
        
        if(args.numinvestors):
            print("**** INVESTORS SELECTED *****", num_marketMakersAgents)
        print("Starting...")

        num_agents = num_noise + num_value + num_momentum_agents + num_mm_agents + num_qlearnersMM + additional_ql + num_randomMM + num_marketMakersAgents + num_persistentMM + 1        
        agent_saved_states = {}
        agent_saved_states['agent_state'] = [None] * num_agents

        fine_tuning_rounds = 0
        finetuning_epsilon = 0
        freeze_n_layers = False
        
        config = {
            "num_mm_agents": num_mm_agents,
            "num_noise": num_noise,
            "num_value": num_value,
            "num_momentum_agents": num_momentum_agents,
            "num_qlearnersMM": num_qlearnersMM,
            "num_randomMM": num_randomMM,
            "num_persistentMM": num_persistentMM,
            "num_marketMakersAgents": num_marketMakersAgents,
            "num_agents": num_agents,
            "num_behaviouralclonners": num_behaviouralclonners,
            "num_adversarial": num_adversarial,
            "turbo_epsilon_clonning": turbo_epsilon,
            "turbo_epsilon_main": main_turbo_epsilon,
            "num_simulations": num_simulations,
            "n_states": n_states,
            "n_actions": n_actions,
            "reduced_states": reduced_states,
            "alpha": alpha,
            "alpha_decay": alpha_decay,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "epsilon_min": epsilon_min,
            "gamma": gamma,
            "force_same_epsilon": force_same_epsilon,
            "qtable_reused": qtable_file,
            "include_hedges": include_hedges,
            "no_inventory": no_inventory,
            "rewardPenalty": rewardPenalty,
            "penaltyMultiplier": penaltyMultiplier,
            "learningRate": learningRate,
            "neuralModel": neuralModel, 
            "keep_inv_cash": args.keep_inv_cash,
            "end_time": args.end_time.strftime('%H:%M:%S'),
            "start_time": args.start_time.strftime('%H:%M:%S'),
            "alphareward": args.alphareward,
            "dql_file_1": args.dql_file_1,
            "dql_file_2": args.dql_file_2,
            "dql_file_3": args.dql_file_3,
            "dql_comp_file_1": args.dql_comp_file_1,
            "finetuning_epsilon": finetuning_epsilon,
            "finetuning_rbuff_mix": finetuning_rbuff_mix,
            "fine_tuning_rounds": fine_tuning_rounds,
            "time": dt.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
            "seed": seed,
            "n_e": n_e,
            "additional_ql": additional_ql,
            "exp_conf": e,
            "optimal_dqls": optimal_dqls,
            "freeze_n_layers": freeze_n_layers,
            "old_replay_buffer": "",
            "apply_old_replay_buffer": args.apply_old_replay_buffer,
            "th_alpha": th_alpha,
            "th_beta": th_beta,
            "th_gamma": th_gamma
        }

        with open(experiment_folder + "config_econf"+str(n_e)+".json", 'w') as outfile:
            json.dump(config, outfile)

        ## RUN ##
        for i in range(num_simulations):
            try:
                del agents
                del oracle
            except:
                pass
            gc.collect()

            simulation_start_time = dt.datetime.now()
            print("\n\n$$$$$$$ SIMULATION NUMBER", i, "of", num_simulations, "$$$$$$$$\n")
            print("Simulation Start Time: {}".format(simulation_start_time))
            print("Configuration seed: {}\n".format(seed))
            
            ########################################################################################################################
            ############################################### AGENTS CONFIG ##########################################################
            ########################################################################################################################

            # Historical date to simulate.
            historical_date = pd.to_datetime(args.historical_date)
            mkt_open = historical_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
            mkt_close = historical_date + pd.to_timedelta(args.end_time.strftime('%H:%M:%S'))
            agent_count, agents, agent_types = 0, [], []

            # Hyperparameters
            symbol = args.ticker
            starting_cash = 10000000  # Cash in this simulator is always in CENTS.

            r_bar = 1e5
            sigma_n = r_bar / 10
            kappa = 1.67e-15
            lambda_a = 7e-11

            # Oracle
            symbols = {symbol: {'r_bar': r_bar,
                                'kappa': 1.67e-16,
                                'sigma_s': 0,
                                'fund_vol': args.fund_vol,
                                'megashock_lambda_a': 2.77778e-18,
                                'megashock_mean': 1e3,
                                'megashock_var': 5e4,
                                'random_state': randomstate}}

            oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

            # 1) Exchange Agent
            stream_history_length = 25000

            agents.extend([ExchangeAgent(id=0,
                                        name="EXCHANGE_AGENT",
                                        type="ExchangeAgent",
                                        mkt_open=mkt_open,
                                        mkt_close=mkt_close,
                                        symbols=[symbol],
                                        log_orders=exchange_log_orders,
                                        pipeline_delay=0,
                                        computation_delay=0,
                                        stream_history=stream_history_length,
                                        book_freq=book_freq,
                                        wide_book=True,
                                        random_state=randomstate)])
            agent_types.extend("ExchangeAgent")
            agent_count += 1

            # 2) Noise Agents
            noise_mkt_open = historical_date + pd.to_timedelta("09:00:00")
            noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")
            agents.extend([NoiseAgent(id=j,
                                    name="NoiseAgent {}".format(j),
                                    type="NoiseAgent",
                                    symbol=symbol,
                                    starting_cash=starting_cash,
                                    wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
                                    log_orders=log_orders,
                                    random_state=randomstate)
                        for j in range(agent_count, agent_count + num_noise)])
            agent_count += num_noise
            agent_types.extend(['NoiseAgent'])

            # 3) Value Agents
            agents.extend([ValueAgent(id=j,
                                    name="Value Agent {}".format(j),
                                    type="ValueAgent",
                                    symbol=symbol,
                                    starting_cash=starting_cash,
                                    sigma_n=sigma_n,
                                    r_bar=r_bar,
                                    kappa=kappa,
                                    lambda_a=lambda_a,
                                    log_orders=log_orders,
                                    random_state=randomstate)
                        for j in range(agent_count, agent_count + num_value)])
            agent_count += num_value
            agent_types.extend(['ValueAgent'])

            # 4) Market Maker Agents
            num_mm_agents = len(mm_params)
            mm_cancel_limit_delay = 50  # 50 nanoseconds

            agents.extend([AdaptiveMarketMakerAgent(id=j,
                                            name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                                            type='AdaptivePOVMarketMakerAgent',
                                            symbol=symbol,
                                            starting_cash=starting_cash,
                                            pov=mm_params[idx][1],
                                            min_order_size=mm_params[idx][4],
                                            window_size=mm_params[idx][0],
                                            num_ticks=mm_params[idx][2],
                                            wake_up_freq=mm_params[idx][3],
                                            cancel_limit_delay=mm_cancel_limit_delay,
                                            skew_beta=args.mm_skew_beta,
                                            level_spacing=args.mm_level_spacing,
                                            spread_alpha=args.mm_spread_alpha,
                                            backstop_quantity=args.mm_backstop_quantity,
                                            log_orders=log_orders,
                                            random_state=randomstate)
                        for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))])
            agent_count += num_mm_agents
            agent_types.extend('POVMarketMakerAgent')
            
            # 5) Momentum Agents
            agents.extend([MomentumAgent(id=j,
                                        name="MOMENTUM_AGENT_{}".format(j),
                                        type="MomentumAgent",
                                        symbol=symbol,
                                        starting_cash=starting_cash,
                                        min_size=1,
                                        max_size=10,
                                        wake_up_freq='20s',
                                        log_orders=log_orders,
                                        random_state=randomstate)
                        for j in range(agent_count, agent_count + num_momentum_agents)])
            agent_count += num_momentum_agents
            agent_types.extend("MomentumAgent")

            # 6) Execution Agent
            trade = True if args.execution_agents else False

            # 7) QL MARKET MAKER AGENTS AND INVESTORS
            num_marketMakers = num_randomMM + num_qlearnersMM + num_persistentMM + additional_ql
            marketMakerIds = []
            marketMakersAgents = []
            marketmakers = []

            for n in range(agent_count, agent_count + num_marketMakers):
                marketmakers.append(n)
                
            for j in range(num_marketMakersAgents):
                marketMakersAgents.append(agent_count + num_marketMakers + j)

            noise_mkt_open = historical_date + pd.to_timedelta("09:00:00")
            noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")
                         
            # DQL MARKET MAKER & CLONERS
            if(num_qlearnersMM > 0):            
                for j in range(num_qlearnersMM):
                    if(j==0):
                        if(main_dqn_last_state != None):
                            agent_saved_states['agent_state'][agent_count] = main_dqn_last_state
                            print("-\|/- ", agent_count, "AGENT STATE UPDATED WITH LAST DATA -\|/-")
                    
                        main_mm_dqn_id = agent_count
                        main_dqn_list.append(main_mm_dqn_id)

                    last_epsilon = False

                    if(j==0 and num_behaviouralclonners):
                        clone = True
                    else:
                        clone = False

                    if(j==0 and num_adversarial):
                        adversarial = True
                    else:
                        adversarial = False
        
                    if agent_saved_states['agent_state'][agent_count] is None:
                        rbuff = False
                        last_active_dql = None
                        current_weights = {0: 0.25, 1:0.25, 5:0.25, 7:0.25}
                        thomson_coefs = {0:(1,1), 1:(1,1), 5:(1,1), 7:(1,1)}
                        tramos = {0: (0,180), 1: (180,360), 5: (360,540), 7: (540,720)}

                        if((loaded_nn_1) and (j == 0)):
                            last_epsilon = 0
                            qweights_1 = loaded_nn_1["nn_weights_1"]
                            qweights_2 = loaded_nn_1["nn_weights_2"]

                            avoid_retrain_agent = avoid_retrain
                            
                            print("||||||||||||||||||\n|||||||||| using existing DQL NN 1...|||||||||\n|||||||||||||||||||||||||", dql_file_1)
                            
                        elif((loaded_nn_2) and (j == 1)):
                            last_epsilon = 0
                            qweights_1 = loaded_nn_2["nn_weights_1"]
                            qweights_2 = loaded_nn_2["nn_weights_2"]

                            avoid_retrain_agent = avoid_retrain_ql2
                            
                            print("||||||||||||||||||\n|||||||||| using existing DQL NN 2...|||||||||\n|||||||||||||||||||||||||", dql_file_2)
                        else:         
                            print("Clean NN")           
                            qweights_1 = False
                            qweights_2 = False

                            avoid_retrain_agent = avoid_retrain
                            
                    else:
                        if(j==0 and use_optimal_dql):
                            print("//\\//\\ BEING OPTIMAL WE RELOAD A NEW NN FOR THE FIRST DQN")
                            qweights_1 = loaded_nn_1["nn_weights_1"]
                            qweights_2 = loaded_nn_1["nn_weights_2"]
                        else:
                            qweights_1 = agent_saved_states['agent_state'][agent_count]['nn_weights_1']
                            qweights_2 = agent_saved_states['agent_state'][agent_count]['nn_weights_2']

                        rbuff = agent_saved_states['agent_state'][agent_count]['rbuff']
                        last_active_dql = agent_saved_states['agent_state'][agent_count]['last_active_dql']
                        thomson_coefs = agent_saved_states['agent_state'][agent_count]['thomson_coefs']
                        current_weights = agent_saved_states['agent_state'][agent_count]['current_weights']
                        tramos = agent_saved_states['agent_state'][agent_count]['tramos']
                        

                        if(j==0): # To adapt the epsilon through the simulations
                            b = int(((num_simulations) / turbo_epsilon_repeats)*0.8)                            
                            try: 
                                a = (i) % (b)
                            except: 
                                a = (i) % 1

                            if(i>=b):
                                c = 0
                            else:
                                try: 
                                    c = 1 - a/(b-1)
                                except: 
                                    c = 0
                            last_epsilon = max(c, epsilon_min)

                        else:                                 
                            if(j==1):
                                last_epsilon = max((num_simulations - (i * turbo_epsilon)) * (1 / num_simulations), epsilon_min) 
                            else:
                                last_epsilon = max((num_simulations - i) * (1 / num_simulations), epsilon_min)             
                    
                        avoid_retrain_agent = agent_saved_states['agent_state'][agent_count]["avoid_retrain"]

                        if(n_e>=0):
                            last_epsilon = 0 

                    agents.extend([DQL_MarketMakerQL(id=agent_count,
                                                    name="MarketMakerDRLQLearning {}".format(agent_count),
                                                    type="MarketMaker",
                                                    symbol=symbol,
                                                    qweights_1=qweights_1,
                                                    qweights_2=qweights_2,
                                                    multi_weight=multi_weight,                                                    
                                                    last_epsilon=last_epsilon,
                                                    epsilon_min=epsilon_min,
                                                    agent_type="QL",
                                                    include_hedges=include_hedges,
                                                    num_exp=i,
                                                    avoid_retrain=avoid_retrain_agent,
                                                    round_exp=round_exp,
                                                    save_results=save_results,
                                                    exp_folder=experiment_folder,
                                                    starting_cash=starting_cash,
                                                    log_orders=log_orders,
                                                    no_inventory=no_inventory,
                                                    force_same_epsilon=force_same_epsilon,
                                                    marketMakersAgents=marketMakersAgents,
                                                    no_qtable_weights=False,
                                                    rewardPenalty=rewardPenalty,
                                                    learningRate=learningRate,
                                                    penaltyMultiplier=penaltyMultiplier,
                                                    neural_model=neuralModel,
                                                    beha_clone=clone,
                                                    marketmakers=marketmakers,
                                                    adversarial=adversarial,     
                                                    reduced_states=reduced_states,  
                                                    keep_inv_cash=args.keep_inv_cash,
                                                    alphareward=args.alphareward,
                                                    states=agent_saved_states['agent_state'][agent_count],
                                                    n_e=n_e,
                                                    main_mm_id=main_mm_dqn_id,
                                                    random_state=randomstate,
                                                    finetuning_epsilon=finetuning_epsilon,
                                                    finetuning_rbuff_mix=finetuning_rbuff_mix,
                                                    fine_tuning_rounds=fine_tuning_rounds,
                                                    rbuff=rbuff,
                                                    max_sims=num_simulations,
                                                    freeze_n_layers=freeze_n_layers,
                                                    apply_old_replay_buffer=old_replay_buffer_file,
                                                    list_optimal_rewards=op_rewards_lists,
                                                    optimal_dqls=optimal_dqls,
                                                    last_active_dql=last_active_dql,
                                                    additional_qls=additional_ql,
                                                    thomson_coefs=thomson_coefs,
                                                    current_weights=current_weights,
                                                    tramos=tramos,
                                                    th_alpha=th_alpha,
                                                    th_beta=th_beta,
                                                    th_gamma=th_gamma)])

                    marketMakerIds.append(agent_count)
                    agent_count += 1

            # Market Maker Random
            if(num_randomMM > 0):

                agents.extend([DQL_MarketMakerQL(id=j,
                                                name="MarketMakerRANDOM {}".format(j),
                                                type="MarketMaker",
                                                symbol=symbol,
                                                qweights_1=False,
                                                qweights_2=False,
                                                last_epsilon=False,
                                                epsilon_min=epsilon_min,
                                                agent_type="RANDOM",
                                                include_hedges=include_hedges,
                                                num_exp=i,
                                                avoid_retrain=False,
                                                round_exp=round_exp,
                                                save_results=save_results,
                                                exp_folder=experiment_folder,
                                                starting_cash=starting_cash,
                                                log_orders=log_orders,
                                                no_inventory=no_inventory,
                                                force_same_epsilon=force_same_epsilon,
                                                marketMakersAgents=marketMakersAgents,
                                                no_qtable_weights=False,
                                                rewardPenalty=rewardPenalty,   
                                                penaltyMultiplier=penaltyMultiplier,  
                                                marketmakers=marketmakers,  
                                                keep_inv_cash=args.keep_inv_cash,
                                                alphareward=args.alphareward,
                                                states=agent_saved_states['agent_state'][agent_count],                                         
                                                random_state=randomstate)
                                            for j in range(agent_count, agent_count + num_randomMM)])
                
                for m in range(agent_count, agent_count + num_randomMM):
                    marketMakerIds.append(m)
                    print("APPEND RANDOM", m)

                agent_count += num_randomMM
                agent_types.extend(['DQL_MarketMakerQL'])

            # Market Maker Persistent
            if(num_persistentMM > 0):

                agents.extend([DQL_MarketMakerQL(id=j,
                                                name="MarketMakerPERSISTENT {}".format(j),
                                                type="MarketMaker",
                                                symbol=symbol,
                                                qweights_1=False,
                                                qweights_2=False,
                                                last_epsilon=False,
                                                epsilon_min=epsilon_min,
                                                agent_type="PERSISTENT",
                                                include_hedges=include_hedges,
                                                num_exp=i,
                                                avoid_retrain=False,
                                                round_exp=round_exp,
                                                save_results=save_results,
                                                exp_folder=experiment_folder,
                                                starting_cash=starting_cash,
                                                log_orders=log_orders,
                                                no_inventory=no_inventory,
                                                force_same_epsilon=force_same_epsilon,
                                                marketMakersAgents=marketMakersAgents,
                                                no_qtable_weights=False,
                                                rewardPenalty=rewardPenalty,
                                                penaltyMultiplier=penaltyMultiplier,
                                                marketmakers=marketmakers,
                                                keep_inv_cash=args.keep_inv_cash,
                                                alphareward=args.alphareward,
                                                states=agent_saved_states['agent_state'][agent_count],
                                                random_state=randomstate)
                                            for j in range(agent_count, agent_count + num_persistentMM)])
                
                for m in range(agent_count, agent_count + num_persistentMM):
                    marketMakerIds.append(m)
                    print("APPEND PERSISTENT", m)

                agent_count += num_persistentMM
                agent_types.extend(['DQL_MarketMakerQL'])

            # Additional QLs (competitors)
            if(additional_ql > 0):
                print("init additional QL:", additional_ql)

                for j in range(agent_count, agent_count + additional_ql):
                    tmp = "_"+str(149 - (j - agent_count) )+"_"
                    file_net = dql_comp_file_1.replace("_149_", tmp)

                    loaded_nn_comp_1 = pickle.load(open(file_net, 'rb'))
                    print(" *** ??? ¿¿¿ *** NETWORK loaded for competitor QL", j, file_net)
                    epsilon = 0
                    epsilon_min = 0

                    last_epsilon = 0
                    qweights_addit_1 = loaded_nn_comp_1["nn_weights_1"]
                    qweights_addit_2 = loaded_nn_comp_1["nn_weights_2"]
                    
                    agents.extend([DQL_MarketMakerQL(id=j,
                                                    name="MarketMakerDRLCOMPETITOR {}".format(agent_count),
                                                    type="MarketMaker",
                                                    symbol=symbol,
                                                    qweights_1=qweights_addit_1,
                                                    qweights_2=qweights_addit_2,
                                                    multi_weight=multi_weight,                                                    
                                                    last_epsilon=0,
                                                    epsilon_min=0,
                                                    agent_type="QL_COMPETITOR",
                                                    include_hedges=include_hedges,
                                                    num_exp=i,
                                                    avoid_retrain=True,
                                                    round_exp=round_exp,
                                                    save_results=save_results,
                                                    exp_folder=experiment_folder,
                                                    starting_cash=starting_cash,
                                                    log_orders=log_orders,
                                                    no_inventory=no_inventory,
                                                    force_same_epsilon=force_same_epsilon,
                                                    marketMakersAgents=marketMakersAgents,
                                                    no_qtable_weights=False,
                                                    rewardPenalty=rewardPenalty,
                                                    learningRate=learningRate,
                                                    penaltyMultiplier=penaltyMultiplier,
                                                    neural_model=neuralModel,
                                                    beha_clone=clone,
                                                    marketmakers=marketmakers,
                                                    adversarial=adversarial,     
                                                    reduced_states=reduced_states,  
                                                    keep_inv_cash=args.keep_inv_cash,
                                                    alphareward=args.alphareward,
                                                    states=agent_saved_states['agent_state'][agent_count],
                                                    n_e=n_e,
                                                    main_mm_id=False,
                                                    random_state=randomstate,
                                                    finetuning_epsilon=0,
                                                    finetuning_rbuff_mix=0,
                                                    fine_tuning_rounds=0)])
                
                for m in range(agent_count, agent_count + additional_ql):
                    marketMakerIds.append(m)
                    print("APPEND ADDITIONAL COMPETITOR QL", m)
                    
                agent_count += additional_ql
                agent_types.extend(['DQL_MarketMakerQL'])

            # Investor Agents
            noise_mkt_open = historical_date + pd.to_timedelta("09:00:00")
            noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")

            agents.extend([MMInvestorAgent(id=j,
                                name="MMInvestorAgent {}".format(j),
                                type="MMInvestorAgent",
                                symbol=symbol,
                                starting_cash=starting_cash,
                                log_orders=log_orders,
                                random_state=randomstate,
                                marketMakerIds=marketMakerIds)
                        for j in range(agent_count, agent_count + num_marketMakersAgents)])
            agent_count += num_marketMakersAgents
            agent_types.extend(['MMInvestorAgent'])

            ########################################################################################################################
            ########################################### KERNEL AND OTHER CONFIG ####################################################
            ########################################################################################################################

            print("MARKET MAKERS IDs", marketMakerIds)
            
            kernel = Kernel("RMSC03 Kernel", random_state=randomstate)

            kernelStartTime = historical_date
            kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')

            defaultComputationDelay = 50  # 50 nanoseconds

            # LATENCY
            latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2**32))
            pairwise = (agent_count, agent_count)

            # All agents sit on line from Seattle to NYC
            nyc_to_seattle_meters = 3866660
            pairwise_distances = util.generate_uniform_random_pairwise_dist_on_line(0.0, nyc_to_seattle_meters, agent_count,
                                                                                random_state=latency_rstate)
            pairwise_latencies = util.meters_to_light_ns(pairwise_distances)

            model_args = {
                'connected': True,
                'min_latency': pairwise_latencies
            }

            latency_model = LatencyModel(latency_model='deterministic',
                                        random_state=latency_rstate,
                                        kwargs=model_args
                                        )
            # KERNEL
           
            agent_saved_states = kernel.runner(agents=agents,
                        startTime=kernelStartTime,
                        stopTime=kernelStopTime,
                        agentLatencyModel=latency_model,
                        defaultComputationDelay=defaultComputationDelay,
                        oracle=oracle,
                        skip_log=True,
                        log_dir=args.log_dir)

            main_dqn_last_state = agent_saved_states['agent_state'][main_mm_dqn_id]

            simulation_end_time = dt.datetime.now()
            print("Simulation End Time: {}".format(simulation_end_time))
            print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
            gc.collect()
            print("*** MEMORY USAGE OF AGENT_SAVED_STATES => ", sys.getsizeof(agent_saved_states))
            print("#### CYCLE FINISHED AT: --> ", dt.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), " #####")

        n_e += 1

    with open(experiment_folder + "main_dqn_list.txt", 'w') as outfile:
        json.dump(main_dqn_list, outfile)
                
if __name__ == "__main__":
    main()