from agent.TradingAgent import TradingAgent
from util.util import log_print
from message.Message import Message
from model.MORLmodel import DQLAgent
from collections import deque 
from statistics import mean, stdev
import gc
import pickle
import random
import pandas as pd
import numpy as np
import time
import torch
import os 
from collections import Counter
import scipy.stats as stats

MAX_LEN_DEQUE_CONTROL = 5  # rounds needed for control
NUMBER_TIMESTEPS_TESTS = 150
T_ROUNDS = [0, 1, 5, 7]
MAX_REW_THRES = 5000000
TIMESTEPS_SIM = 720

class DQL_MarketMakerQL(TradingAgent):

    def __init__(self, id, name, type, symbol, starting_cash, marketMakersAgents, epsilon_min, random_state, marketmakers, qweights_1, qweights_2, pareto_round=-1, epsilonBuy=0.01, epsilonSell=0.01, last_epsilon=False,
                      multi_weight=0.5, wake_up_freq='1s', agent_type="DRL-QL", subscribe_freq=10e9, exp_folder=None, num_exp=1, no_inventory=False,
                      save_results=True, log_orders=False, round_exp=0, force_same_epsilon=False, no_qtable_weights=False,
                      avoid_retrain=False, include_hedges=False, rewardPenalty=0, learningRate=0.01, penaltyMultiplier=0.01, neural_model=0,
                      beha_clone=False, adversarial=False, reduced_states=False, 
                      keep_inv_cash=False, states=False, alphareward=1, force_hedging_limit=False,
                      n_e=False, main_mm_id=False, loaded_mod_rt=False, 
                      iteration=0, finetuning_epsilon=0.5,
                      fine_tuning_rounds=15, rbuff=False, max_sims=150,
                      freeze_n_layers=False, apply_old_replay_buffer=False,
                      list_optimal_rewards=False, optimal_dqls=False,
                      last_active_dql=None, additional_qls=0,
                      last_control_rounds=deque(maxlen=MAX_LEN_DEQUE_CONTROL),
                      current_weights=False, thomson_coefs=False,
                      sampling_every_rnd=3, tramos={},
                      th_alpha=1, th_beta=1, th_gamma=0.85):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)

        self.device = torch.device("cuda")

        self.last_active_dql = last_active_dql
        self.list_of_changepoints = []
        self.additional_qls = additional_qls
        self.trues = []
        self.last_control_rounds = last_control_rounds
        
        self.perform_agent_test = False

        self.iteration = iteration         
        self.current_weights = current_weights
        self.thomson_coefs = thomson_coefs
        self.ag_selected = 0
        self.sampling_every_rnd = sampling_every_rnd
        self.iter_thomsom = 0
        self.current_max_iter_thomson = 0
        self.th_alpha = th_alpha
        self.th_beta = th_beta
        self.th_gamma = th_gamma

        self.tramos = tramos
        
        if(num_exp % self.sampling_every_rnd == 0):
            self.perform_agent_test = True
            self.tests_performed = {}
            self.current_test_agent = 0
            for t in T_ROUNDS:
                self.tests_performed[t] = []
        
        self.in_finetuning = False
        self.last_epsilon_finetuning = False
        self.finetuning_epsilon = finetuning_epsilon
        self.fine_tuning_rounds = fine_tuning_rounds
        self.exp_replay_finetun = deque(maxlen=1000000)
        self.list_optimal_rewards = list_optimal_rewards
        self.symbol = symbol
        self.wake_up_freq = wake_up_freq
        self.subscribe_freq = subscribe_freq       
        self.subscription_requested = False
        self.log_orders = log_orders
        self.no_inventory = no_inventory
        self.rewardPenalty = rewardPenalty
        self.penaltyMultiplier = penaltyMultiplier    
        self.max_sims = max_sims
        self.marketMakersAgents = marketMakersAgents
        self.marketMakersAgentsChecked = len(marketMakersAgents)       
        self.marketMakerSpread = [-1, -1]
        self.mid = -1
        self.lastMid = -1
        self.spread = -1
        self.lastSpread = -1      
        self.epsilon = 0        
        self.alphareward = float(alphareward)
        self.multi_weight = float(multi_weight)
        self.qweights_1 = qweights_1
        self.qweights_2 = qweights_2
        self.pareto_round = pareto_round
        
        print("MULTI WEIGHT APPLIED ->", self.multi_weight)

        self.force_hedging_limit = force_hedging_limit
        self.max_limit_force_hedging = 1000000
        self.apply_old_replay_buffer = apply_old_replay_buffer
        self.ql_mm_adversarials = 0        
        self.beha_clone = beha_clone
        self.adversarial = adversarial
        self.adversarial_reward = 0

        if(not self.beha_clone and not self.adversarial):
            self.infoagents_updated = True
        else:
            self.infoagents_updated = False

        self.competitorMarketMakerSpread = {}
        self.competitorMarketMakerUpdated = {} 
        self.price_streamed = False
        self.no_newmarketdata = True       
        self.marketmakers = marketmakers

        for x in self.marketmakers:
            if(x != self.id):
                self.competitorMarketMakerSpread[x] = []
                self.competitorMarketMakerUpdated[x] = False 

        if(self.beha_clone):
            if(self.adversarial):
                raise("ERROR -> cloner and adversary selected. Pick only one")
            
            self.ql_mm_adversarials = 1
        
        self.n_e = n_e
        self.main_mm_id = main_mm_id
        self.loaded_mod_rt = loaded_mod_rt
        
        if(num_exp == 0 or not keep_inv_cash):
            self.mark_to_value = 10000000
            self.last_mark_to_value = 10000000
            self.init_cash = 10000000
            self.inventory = 0
            self.lastInventory = 0
        else:
            self.mark_to_value = states["mark_to_value"]
            self.last_mark_to_value = states["mark_to_value"]
            self.init_cash = states["cash"]
            self.inventory = states["inventory"]  
            self.lastInventory = states["inventory"]  

        self.epsilonBuy = epsilonBuy
        self.epsilonSell = epsilonSell
        self.force_same_epsilon = force_same_epsilon

        if(reduced_states): 
            self.state_size = 2 + 2 * (self.ql_mm_adversarials) 
        else:
            self.state_size = 8 + 2 * (self.ql_mm_adversarials)
        
        self.hedges_ranges = [0]
        self.include_hedges = include_hedges
        self.action_list = []

        if(force_same_epsilon):
            self.epsilon_ranges = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
            if(self.include_hedges and not self.force_hedging_limit):
                self.hedges_ranges = [0, 0.25, 0.5, 0.75, 1]  

            self.action_size = len(self.epsilon_ranges) * len(self.hedges_ranges)

            for b in range(0, len(self.epsilon_ranges)):
                for h in range(0, len(self.hedges_ranges)):
                    self.action_list.append([self.epsilon_ranges[b], self.hedges_ranges[h]])
        else:        
            self.epsilon_ranges = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
            if(self.include_hedges and not self.force_hedging_limit):
                self.hedges_ranges = [0, 0.25, 0.5, 0.75, 1]        

            self.action_size = len(self.epsilon_ranges) * len(self.epsilon_ranges) * len(self.hedges_ranges)
            
            for b in range(0, len(self.epsilon_ranges)):
                for s in range(0, len(self.epsilon_ranges)):
                    for h in range(0, len(self.hedges_ranges)):
                        self.action_list.append([self.epsilon_ranges[b], self.epsilon_ranges[s], self.hedges_ranges[h]])

        print("\n##################\nAGENT TYPE:", agent_type)
        if(self.beha_clone):
            print("--$$$$$$$ BEHAVIORAL CLONER $$$$$$----", self.beha_clone)
        if(self.adversarial):
            print("--$$$$$$$ ADVERSARIAL $$$$$$----", self.adversarial)
        print("FORCE SAME EPSILON:", self.force_same_epsilon)
        print('self.epsilonBuy ', self.epsilonBuy)
        print('self.epsilonSell ', self.epsilonSell)
        print("self.epsilon_range:", self.epsilon_ranges)
        print("self.save_results:", save_results)
        print("self.id:", self.id)
        print("MM Agents", marketMakersAgents)
        print("STATE SPACE:", self.state_size)
        print("ACTION SPACE:", len(self.action_list))
        print("ALPHA REWARD:", self.alphareward)
        print("Pareto round:", self.pareto_round)
        print("multi weight:", self.multi_weight)
        print("th_alpha", th_alpha)
        print("th_beta", th_beta)
        print("th_gamma", th_gamma)
        
        if(num_exp % 10 == 0):
            os.system("drop_cache")
        
        self.exp_folder = exp_folder
        self.buy = 0 
        self.sell = 0
        self.buyQuantity = 0
        self.sellQuantity = 0
        self.lastStepVolume = 0
        self.firstTime = True
        self.saveData = save_results
        self.num_exp = num_exp
        self.historicData = []
        self.rewardsLog = []  
        self.splitted_rewards = []  
        self.invRewardsLenght = 10
        self.invRewards = deque(maxlen=self.invRewardsLenght)
        self.lastInventories = deque(maxlen=10)
        self.lastInventories.append(0)
        self.midprices = deque(maxlen=20)
        self.maxinvs = deque(maxlen=100)        
        self.state = 0
        self.action = 0
        self.accumulatedReward = 0
        self.qlastkind = None
        self.last_spread = None
        self.no_qtable_weights = no_qtable_weights                       
        self.agent_type = agent_type        
        self.fixedAction = random.randint(0, len(self.action_list)-1)
        self.highinvexplored = False
        self.lowinvexplored = False
        self.inv_threshold = 1
        self.explo_order = random.choice([0, 1])

        self.avoid_retrain = avoid_retrain
                        
        print("Avoid retrain? --> ", avoid_retrain)        
        print("Loading scaler...")        
        if(reduced_states):
            self.mix_max_scaler = pickle.load(open("/obj/scalers/standard_scaler_invandspread.pkl", 'rb'))     
        else:
            self.mix_max_scaler = pickle.load(open("/obj/scalers/standard_scaler_trained_asymetric_optimized.pkl", 'rb'))   
        
        self.round_exp = round_exp
        self.last_state = []
        self.last_action = None
        self.last_qaction = 0

        self.hedge_cost = 0

        self.bid = 0
        self.ask = 0
        self.reduced_states = reduced_states
        
        if(self.agent_type == "QL" or self.agent_type == "QL_COMPETITOR"):   
            self.epsilon = 0.99
            self.epsilon_decay = 1
            self.epsilon_min = epsilon_min
            self.training_batch = 200
            self.batch_size = 1024
            
            if(self.avoid_retrain):
                self.epsilon = 0
                print("\n@@@@@@@@@@@@@ @@@@@@@@@@@ AVOID_RETRAIN -> self epsilon forced to", self.epsilon)
            elif(last_epsilon is not False):                
                self.epsilon = last_epsilon
                print("\n@@@@@@@@@@@@@ @@@@@@@@@@@ self epsilon forced to", self.epsilon)
            else:
                print("@@@@@@@@@@@@@ @@@@@@@@@@@ LAST EPSILON TO", self.epsilon)
            
            if(optimal_dqls):
                self.optimal_dqls = {}               
                inner_id = "E"+str(n_e)+"_"+ str(self.round_exp)+"_"+str(self.id)

                print("Loading Different optimals NN + AGENTS for the changepoint setup...")
            
                for x in optimal_dqls.keys():
                    self.optimal_dqls[x] = {}
                    self.optimal_dqls[x]["NN"] = pickle.load(open(optimal_dqls[x], 'rb'))
                    self.optimal_dqls[x]["AGENT"] = DQLAgent(action_size=self.action_size, 
                                        batch=self.batch_size,
                                        state_size=self.state_size,
                                        qweights_1=self.optimal_dqls[x]["NN"]["nn_weights_1"],
                                        qweights_2=self.optimal_dqls[x]["NN"]["nn_weights_2"],
                                        learning_rate=learningRate, 
                                        neural_model=neural_model,
                                        agent=str(self.id),
                                        id=inner_id,
                                        path=self.exp_folder,
                                        round_exp=self.round_exp,
                                        multi_weight=multi_weight,
                                        freeze_n_layers=freeze_n_layers)
                    
                self.DQLAgent = self.optimal_dqls[0]["AGENT"]

                print("####### DRL MM AGENT RANDOM STATE:", random_state.seed())
                try: 
                    self.DQLAgent.q_network_1.summary()
                except: 
                    print("ERROR -> NO SUMMARY")

                if(self.agent_type == "QL" and self.num_exp > 0):
                    if(rbuff):
                        self.DQLAgent.loadBuffer(rbuff)
                    else:
                        self.DQLAgent.loadBuffer()
            else:
                self.DQLAgent = DQLAgent(action_size=self.action_size, 
                                    batch=self.batch_size,
                                    state_size=self.state_size,
                                    qweights_1=self.qweights_1,
                                    qweights_2=self.qweights_2,
                                    learning_rate=learningRate, 
                                    neural_model=neural_model,
                                    agent=str(self.id),
                                    id="E"+str(n_e)+"_"+ str(self.round_exp)+"_"+str(self.id),
                                    path=self.exp_folder,
                                    round_exp=self.round_exp,
                                    multi_weight=multi_weight,
                                    freeze_n_layers=freeze_n_layers)
        
    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)        

        if (msg.body['msg'] == 'PRICE_UPDATED'):
            if(self.infoagents_updated == False and (self.beha_clone or self.adversarial)):                
                if(msg.body["sender"] in self.competitorMarketMakerSpread):
                    if(self.beha_clone):
                        self.competitorMarketMakerSpread[msg.body["sender"]] = msg.body['epsilons_bs']                              
                    elif(self.adversarial):
                        self.competitorMarketMakerSpread[msg.body["sender"]] = msg.body['reward']   
                        if(msg.body["sender"] == 124):
                            self.adversarial_reward = msg.body['reward']
                    else:
                        raise("ERROR -> adversarial or beha error!")                       

                    self.competitorMarketMakerUpdated[msg.body["sender"]] = True

                    self.infoagents_updated = True                    
                    for i in self.competitorMarketMakerUpdated:
                        if(self.competitorMarketMakerUpdated[i] == False):                   
                            self.infoagents_updated = False
            
        elif msg.body['msg'] == 'MARKET_DATA':
            if ((len(self.marketMakersAgents) == self.marketMakersAgentsChecked)):
                if(self.no_newmarketdata):
                    self.iteration = self.iteration + 1                
                    self.lastMid = self.mid                                     

                    bid = msg.body['bids']
                    ask = msg.body['asks']

                    if bid and ask:
                        self.mid = (ask[0][0] + bid[0][0]) / 2
                        self.lastSpread = self.spread
                        self.spread = abs(ask[0][0] - bid[0][0]) / 2
                        self.bid = bid[0][0]
                        self.ask = ask[0][0]
                    else:
                        self.mid = msg.body['last_transaction']
                        if self.spread < 0:
                            self.spread = 0
                            self.lastSpread = self.spread
                    
                    self.no_newmarketdata = False
                    self.midprices.append(self.mid)  
              
                if(self.infoagents_updated):                   
                    reward_1 = 0
                    reward_2 = 0
                    reward_multi = 0
                    
                    if self.firstTime:
                        self.firstTime = False
                    else:
                        self.mark_to_value = self.holdings["CASH"] + (self.inventory * self.mid)
   
                        reward_1 = self.getRewardAlternative("obj1b", option=0, alphareward=self.alphareward)
                        reward_2 = self.getRewardAlternative("obj2", option=0, alphareward=self.alphareward)
                        
                        reward_multi = reward_1 * self.multi_weight + reward_2 * (1 - self.multi_weight)    
                        self.accumulatedReward += reward_multi
                        self.last_mark_to_value = self.mark_to_value

                    newState = self.getState_ampliado(self.reduced_states)
                    self.saveRewardsLog([reward_1, reward_2, reward_multi])
                
                    action = self.getActions_DQL(newState, styleQL="classic")
                    self.marketMakerSpread = action[0]
                    self.hedgedEpsilon = action[2]
                    self.epsilons_bs = action[3]

                    if(self.agent_type == "QL"):
                        if(len(self.last_state) > 0):
                            info = [self.competitorMarketMakerSpread, self.buyQuantity, self.sellQuantity, self.inventory]
                            self.DQLAgent.store(self.last_state, self.last_qaction, [reward_1, reward_2], newState, False, info) 

                        tmp = self.detect_changepoint()     
                                           
                        if(not self.avoid_retrain):
                            if(self.iteration % self.training_batch == 0): 
                                print("retraining ", self.id, self.agent_type, self.avoid_retrain)                           
                                self.DQLAgent.retrain(apply_old_replay_buffer=self.apply_old_replay_buffer)
                                self.DQLAgent.align_target_model()

                        if(self.perform_agent_test):
                            self.tests_performed[self.current_test_agent].append(reward_multi)
                            test_agent_round = int(self.iteration / NUMBER_TIMESTEPS_TESTS)                             

                            if(test_agent_round < len(T_ROUNDS)):
                                self.DQLAgent = self.optimal_dqls[T_ROUNDS[test_agent_round]]["AGENT"]
                                self.current_test_agent = T_ROUNDS[test_agent_round]                    
                            else:
                                self.perform_agent_test = False    
                                winner = None
                                tmp_win = -1e10                 

                                for t in T_ROUNDS: 
                                    r = np.sum(self.tests_performed[t])
                                    if (r > tmp_win):
                                        winner = t
                                        tmp_win = r                               
                                                                                      
                                muestras = [np.random.beta(a, b) for a, b in self.thomson_coefs.values()]
                                age_sel = np.argmax(muestras)
                                es_ganador = T_ROUNDS[age_sel] == winner
                                
                                for z in T_ROUNDS:
                                    a, b = self.thomson_coefs[z]
                                    if(z == T_ROUNDS[age_sel]):
                                        if(z == winner):    
                                            a = self.th_gamma * a + self.th_alpha
                                        else:               
                                            b = self.th_gamma * b + self.th_beta
                                    else:
                                        a = self.th_gamma * a
                                        b = self.th_gamma * b
                                    
                                    self.thomson_coefs[z] = (a, b)

                                suma_total_pesos = np.sum([a / (a + b) for a, b in self.thomson_coefs.values()])

                                for c in self.current_weights.keys():
                                    self.current_weights[c] = (self.thomson_coefs[c][0] / (self.thomson_coefs[c][0] + self.thomson_coefs[c][1])) / suma_total_pesos

                                test_log = pd.DataFrame([[self.n_e, self.num_exp, self.additional_qls, self.tests_performed, self.current_weights, self.thomson_coefs, winner, es_ganador]])
                                test_log.to_csv(self.exp_folder + '_THOMSON_PERFORMED'+self.round_exp+'.csv', index=False, mode='a', header=False, sep="\t")   

                                self.iter_thomsom = 0                                
                                inicio = 0
        
                                for m in self.current_weights.keys():
                                    self.tramos[m] = (inicio, round(inicio + TIMESTEPS_SIM * self.current_weights[m]))
                                    inicio += round(TIMESTEPS_SIM * self.current_weights[m])
                        else:
                            for m in self.tramos.keys():
                                if(self.tramos[m][0] <= self.iter_thomsom < self.tramos[m][1]):
                                    self.DQLAgent = self.optimal_dqls[m]["AGENT"]   
                                    self.ag_selected = m
                            self.iter_thomsom += 1

                    if(self.include_hedges or self.force_hedging_limit):
                        if(not self.force_hedging_limit):
                            self.inventory_hedged = self.lastInventory * self.hedgedEpsilon
                        else:                            
                            if(abs(self.lastInventory) >= self.max_limit_force_hedging):              
                                self.inventory_hedged = self.lastInventory * 1
                                print("Force hedged! ->", self.lastInventory)
                            else:
                                self.inventory_hedged = 0
                        
                        self.hedge_cost = abs(self.inventory_hedged * (self.spread))

                        if(self.inventory_hedged > 0):
                            self.holdings['CASH'] += self.inventory_hedged * (self.mid - self.spread)
                        else:
                            self.holdings['CASH'] += self.inventory_hedged * (self.mid + self.spread)
                        
                        self.inventory = self.inventory - self.inventory_hedged 

                    self.buy = 0
                    self.sell = 0
                    self.buyQuantity = 0
                    self.sellQuantity = 0
                    self.lastStepVolume = 0
                    self.lastInventory = self.inventory
                    self.last_state = newState
                    self.last_qaction = action[1]
                    self.last_spread = self.spread
                    self.marketMakersAgentsChecked = 0
                    self.lastInventories.append(self.inventory)
                    self.lastInventoryMean = np.mean(self.lastInventories)
                    
                    if(self.lastInventory > self.inv_threshold):
                        self.highinvexplored = True
                    elif(self.lastInventory < -self.inv_threshold):
                        self.lowinvexplored = False

                    for i in (self.marketMakersAgents + self.marketmakers):
                        if(i != self.id):
                            self.sendMessage(i, Message({
                                                        "msg": "PRICE_UPDATED",
                                                        "symbol": self.symbol,
                                                        "marketMakerSpread": self.marketMakerSpread,
                                                        "midPrice": self.mid,
                                                        "sender": self.id,
                                                        "state": newState,
                                                        "action": action,
                                                        "agent_type": self.agent_type,
                                                        "id": self.id,
                                                        "epsilons_bs": self.epsilons_bs,
                                                        "reward": [reward_1, reward_2, reward_multi],
                                                        "spread": self.spread
                                                        }))                 

                    self.no_newmarketdata = True

                    if(self.beha_clone or self.adversarial):
                        for x in self.competitorMarketMakerUpdated:
                            self.competitorMarketMakerUpdated[x] = False
                    
                        self.infoagents_updated = False

        elif msg.body['msg'] == 'NOT_TRADE':
            self.marketMakersAgentsChecked += 1

        elif msg.body['msg'] == 'MARKET_MAKER_ORDER':
            self.marketMakersAgentsChecked += 1

            buy = msg.body['buy'] 
            quantity = msg.body['quantity']
            agent = msg.body['agent']

            self.lastStepVolume += quantity

            if buy:
                self.inventory -= quantity                 
                self.holdings['CASH'] += (self.mid + self.marketMakerSpread[1]) * quantity
                agent.holdings['CASH'] -= (self.mid + self.marketMakerSpread[1]) * quantity

                if self.symbol in agent.holdings:
                    agent.holdings[self.symbol] += quantity
                else:
                    agent.holdings[self.symbol] = quantity
                self.sell += 1
                self.sellQuantity += quantity

            else:
                self.inventory += quantity                
                self.holdings['CASH'] -= (self.mid - self.marketMakerSpread[0]) * quantity
                agent.holdings['CASH'] += (self.mid - self.marketMakerSpread[0]) * quantity

                if self.symbol in agent.holdings:
                    agent.holdings[self.symbol] -= quantity
                else:
                    agent.holdings[self.symbol] = -1 * quantity

                self.buy += 1
                self.buyQuantity += quantity
                spreadOp = self.marketMakerSpread[0]

    def saveRewardsLog(self, reward):
        rewardlog = []
        rewardlog.append(self.num_exp)
        rewardlog.append(self.iteration)
        rewardlog.append(self.last_state)
        rewardlog.append(self.last_action)
        rewardlog.append(reward)
        rewardlog.append(self.accumulatedReward)
        rewardlog.append(self.qlastkind)
        rewardlog.append(self.splitted_rewards)

        rewardlog.append(self.mark_to_value)
        rewardlog.append(self.holdings["CASH"])
        rewardlog.append(self.epsilon)
        rewardlog.append(self.buyQuantity)
        rewardlog.append(self.sellQuantity)
        rewardlog.append(self.lastInventory)
        rewardlog.append(self.spread)
        rewardlog.append(self.in_finetuning)
        
        self.rewardsLog.append(rewardlog)
        
    def getState_ampliado(self, reduced_states=False):
        state = []
        
        if(reduced_states):
            state.append(self.lastInventory)
            state.append(self.spread)
        else:
            state.append(self.buyQuantity)
            state.append(self.sellQuantity)
            state.append(self.inventory)
            state.append(self.lastInventory)
            state.append(self.mid - self.lastMid)
            state.append(self.spread)
            state.append(self.lastSpread)
            state.append(self.lastStepVolume)

        if self.saveData:
            statesave = state.copy()
            statesave.append(self.mid)
            statesave.append(self.bid)
            statesave.append(self.ask)
            statesave.append(self.multi_weight)
            statesave.append(self.num_exp)
            statesave.append(self.round_exp)
            statesave.append(self.pareto_round)
            statesave.append(self.in_finetuning)
            statesave.append(self.ag_selected)
            self.historicData.append(statesave)

        state = self.mix_max_scaler.transform([state])

        if(self.beha_clone):
            epsb = 1
            epss = 1

            for v in self.competitorMarketMakerSpread:
                if epsb > (self.competitorMarketMakerSpread[v][0]):
                    epsb = (self.competitorMarketMakerSpread[v][0])
                if epss > (self.competitorMarketMakerSpread[v][1]):
                    epss = (self.competitorMarketMakerSpread[v][1])
            
            state = np.array([state[0].tolist()+[epsb, epss]])
      
        return state
 
    def getActions_DQL(self, newState, styleQL="classic"):
        rand_it = self.random_state.rand()        
        
        # 1- PERSISTENT ACTION
        if(self.agent_type == "PERSISTENT"):
            action = self.fixedAction

            if(self.force_same_epsilon):                
                self.epsilonBuy = self.getValueAction(action)[0]
                self.epsilonSell = self.getValueAction(action)[0]
                self.epsilonHedge = 0.5
                self.qlastkind = "persistent"            
            else:
                self.epsilonBuy = self.getValueAction(action)[0]
                self.epsilonSell = self.getValueAction(action)[1]
                self.epsilonHedge = 0.5

            self.qlastkind = "persistent"
               
        # 2- RANDOM ACTION  
        elif((self.agent_type == "RANDOM")):            
            if(self.force_same_epsilon):
                action = random.randint(0, len(self.epsilon_ranges)-1) 
                self.epsilonBuy = self.getValueAction(action)[0]
                self.epsilonSell = self.getValueAction(action)[0]
                self.epsilonHedge = self.getValueAction(action)[1] 
            else:
                action = random.randint(0, len(self.action_list)-1) 
                self.epsilonBuy = self.getValueAction(action)[0]
                self.epsilonSell = self.getValueAction(action)[1]
                self.epsilonHedge = self.getValueAction(action)[2]
            
            self.qlastkind = "random"

        elif((self.agent_type == "QL")):  
            if(rand_it < self.epsilon):
                if(styleQL == "classic"):                    
                    action = random.randint(0, len(self.action_list)-1) 
                    self.epsilonBuy = self.getValueAction(action)[0]
                    self.epsilonSell = self.getValueAction(action)[1]
                    self.epsilonHedge = self.getValueAction(action)[2]
                    self.qlastkind = "random"
            else:
                bestAction = self.DQLAgent.predict(newState)                                    
                action = bestAction                
                self.epsilonBuy = self.getValueAction(action)[0]
                self.epsilonSell = self.getValueAction(action)[1]
                self.epsilonHedge = self.getValueAction(action)[2]
                self.qlastkind = "best"            

        elif((self.agent_type == "QL_COMPETITOR")):
            bestAction = self.DQLAgent.predict(newState)                                    
            action = bestAction                
            self.epsilonBuy = self.getValueAction(action)[0]
            self.epsilonSell = self.getValueAction(action)[1]
            self.epsilonHedge = self.getValueAction(action)[2]
            self.qlastkind = "best" 
        else:
            raise("ERROR IN ACTION")
      
        buySpread = round(self.spread + (self.spread * self.epsilonBuy), 2)
        sellSpread = round(self.spread + (self.spread * self.epsilonSell), 2)
        self.last_action = [self.epsilonBuy, self.epsilonSell, buySpread, sellSpread, self.epsilonHedge]

        return [buySpread, sellSpread], action, self.epsilonHedge, [self.epsilonBuy, self.epsilonSell]

    def getValueAction(self, action):
        return(self.action_list[action])

    def getRewardAlternative(self, penaltyFuction=0, option=0, alphareward=1):
        rew = self.mark_to_value - self.last_mark_to_value  
        
        if(self.adversarial):
            return rew - self.adversarial_reward
        
        elif option == 0:
            if penaltyFuction == 0:            
                return rew             

            elif penaltyFuction == "obj1b":
                spreadLnL = abs(self.buyQuantity) * (self.marketMakerSpread[0]) + abs(self.sellQuantity) * (self.marketMakerSpread[1])   
                inventoryLnL = self.lastInventory * (self.mid - self.lastMid) 
                rew = spreadLnL + inventoryLnL - self.hedge_cost  
                return rew

            elif penaltyFuction == "obj2":
                alpha = 5
                rew = - alpha * abs(self.lastInventory) - self.hedge_cost                
                return rew

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def kernelStopping(self):
        surplus = 0
        holdings = 0
        
        if self.inventory > 0:
            holdings += self.inventory * (self.mid)
        else:
            holdings += self.inventory * (self.mid)

        if(self.agent_type == "QL"):
            if(not self.avoid_retrain):                
                print("Final retraining  ", self.id, self.agent_type, self.avoid_retrain)                           
                self.DQLAgent.retrain()
                self.DQLAgent.align_target_model()

        print('Final holdings for ', self.name, ': { ', self.symbol, ' ', self.inventory, ', CASH: ', round(self.holdings['CASH']), ' }. Marked to market: ', round(holdings + self.holdings['CASH']))

        if(self.agent_type == "QL"):

            qw1 = self.DQLAgent.q_network_1.state_dict() 
            qw2 = self.DQLAgent.q_network_2.state_dict()
           
            self.updateAgentState({'nn_weights_1': qw1, 
                                'nn_weights_2': qw2, 
                                'reward': self.accumulatedReward, 
                                'last_epsilon': self.epsilon, 
                                'mean_loss_list_1':self.DQLAgent.mean_loss_list_1, 
                                'mean_loss_list_2':self.DQLAgent.mean_loss_list_2, 
                                'avoid_retrain':self.avoid_retrain, 
                                'save_buffer':True,  
                                "cash":round(self.holdings['CASH']), 
                                "inventory":self.inventory, 
                                "mark_to_value": self.mark_to_value,
                                "global_iteration": self.iteration,
                                "rbuff": self.DQLAgent.experience_replay,
                                "last_active_dql": self.last_active_dql,
                                "last_control_rounds": self.last_control_rounds if not self.perform_agent_test else deque(maxlen=MAX_LEN_DEQUE_CONTROL),
                                "perform_agent_test": self.perform_agent_test,
                                "current_weights":self.current_weights,
                                "thomson_coefs": self.thomson_coefs,
                                "tramos": self.tramos
                                })
                        
            print("LOG:--> TOTAL ITERATIONS", self.id, self.iteration)

        else:
            self.updateAgentState({'nn_weights_1': None, 'nn_weights_2': None, 'reward': self.accumulatedReward, 'last_epsilon': None, 'mean_loss_list_1': None, 'mean_loss_list_2': None, "cash":round(self.holdings['CASH']), "inventory":self.inventory, "mark_to_value": self.mark_to_value})

        if((self.num_exp == (self.max_sims - 1)) or self.num_exp <= 1):
            if(self.agent_type == "QL"):            
                print("@@@@ SAVING REPLAY BUFFER AT LAST ROUND " + str(self.num_exp) + " @@@@", self.id)
                self.DQLAgent.saveBuffer() 
            
            if self.saveData:
                pandas = pd.DataFrame(self.historicData)
                pandas.to_csv(self.exp_folder + self.agent_type + '_' + self.round_exp + '_HISTORIC_PAPER_STATES' + str(self.id) + '_E' + str(self.n_e) + '.bz2', index=False, mode='a', header=False, compression="bz2")

                pandas = pd.DataFrame([[self.num_exp, self.accumulatedReward]])
                pandas.to_csv(self.exp_folder + self.agent_type + '_ACCUMREWARDS_' + self.round_exp + '_' + str(self.id) + '_E' + str(self.n_e) + '.bz2', index=False, mode='a', header=False, compression="bz2")
                print("__saving FINAL: _ACCUMREWARDS_, _HISTORIC_PAPER_STATES")

                pandas = pd.DataFrame(self.rewardsLog)
                pandas.to_csv(self.exp_folder + self.agent_type + '_' + self.round_exp + '_REWARDS_PAPER_STATES' + str(self.id) + '_E' + str(self.n_e) + '.bz2', index=False, mode='a', header=False, compression="bz2")
                print("__saving FINAL: _REWARDS_PAPER_STATES")

            del pandas   
            
        if((self.saveData) and (self.agent_type == "QL")):
            nn_dict = {"nn_weights_1": qw1, "nn_weights_2": qw2, "epsilon": self.epsilon, "epsilon_decay": self.epsilon_decay, "mean_loss_list_2": self.DQLAgent.mean_loss_list_2, "mean_loss_list_2": self.DQLAgent.mean_loss_list_2}
            pickle.dump(nn_dict, open(self.exp_folder + self.agent_type + "_TORCH_nnweights_" + self.round_exp + "_" + str(self.pareto_round) + "_" + str(self.num_exp) + "_" + str(self.id) + '_E' + str(self.n_e) + ".pkl", "wb"))

            pandas = pd.DataFrame(self.rewardsLog)
            pandas.to_csv(self.exp_folder + self.agent_type + '_' + self.round_exp + '_REWARDS_PAPER_STATES' + str(self.id) + '_E' + str(self.n_e) + '.bz2', index=False, mode='a', header=False, compression="bz2")
            print("__saving: _REWARDS_PAPER_STATES")
            del nn_dict

        print("LAST CONTROL ROUNDS->", self.last_control_rounds)
        gc.collect()
        
    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)

        if not self.subscription_requested:
            super().requestDataSubscription(self.symbol, levels=1, freq=10e9)
            self.subscription_requested = True
            self.state = 'AWAITING_MARKET_DATA'

    def from_pricespread_to_epsilon(self, pricespread, spread):
        return (pricespread - spread) / spread

    def detect_changepoint(self):
        return False