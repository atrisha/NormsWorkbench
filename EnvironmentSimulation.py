'''
Created on 19 Aug 2022

@author: Atrisha
'''
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from norm_objects import *
import constants
import numpy as np
from scipy.interpolate import interp1d
from anaconda_project.internal.conda_api import result
import utils

class StaticSimulation:
    
    def test_static(self):
        fig, ax = plt.subplots(1, 1)
        
        constants.c = 0.0005
        cmap = cm.coolwarm
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
        #all_d_samples = np.linspace(0,0.95,10)
        all_d_samples = [0,0.5,0.9]
        max_time_steps = 100
        colors = cm.coolwarm(np.linspace(0,1,len(all_d_samples)))
        results = dict()
        for d_idx,d_sample in enumerate(all_d_samples):
            
            for group_id in np.arange(100):
                constants.d = d_sample
                imp_games_played,time_idx = 0,0
                player_list = {i:Player(i) for i in np.arange(constants.players_per_group)}
                for pidx,p in player_list.items():
                    punisher_prop_belief = constants.punisher_prop
                    p.imp_game_utility = (0.5/(1-constants.discount_factor))*(((2*punisher_prop_belief-1)*constants.R)-(punisher_prop_belief*constants.c))
                    p.silly_game_utility = 0
                    p.signalling_cost = 0
                
                time_period_results = {0:1}
                
                for time_period in np.arange(max_time_steps):
                    
                    #print(time_period,'players:'+str(len(player_list)),'utils:'+str(np.mean([x.utility for x in player_list.values()]))+'-'+str(np.std([x.utility for x in player_list.values()])))
                    which_interaction_is_important = np.random.choice(np.arange(int(1/(1-constants.d))))
                    for interaction_idx in np.arange(int(1/(1-constants.d))):
                        for player_id in list(player_list.keys()):
                            if player_list[player_id].imp_game_utility + player_list[player_id].signalling_cost < 0:
                                del player_list[player_id]
                        if len(player_list) < 2:
                            continue
                        
                        #env_punisher_sample = constants.punisher_prop*len(player_list)
                        punishers = np.random.choice([True,False],size=len(player_list),p=[constants.punisher_prop,1-constants.punisher_prop])
                        env_punisher_sample = sum(punishers)
                        for p in player_list.values():
                            p.bayes_update_punishers((env_punisher_sample,len(player_list)-env_punisher_sample))
                            punisher_prop_belief = p.sample_punisher_basedon_belief()
                            p.imp_game_utility = (0.5/(1-constants.discount_factor))*(((2*punisher_prop_belief-1)*constants.R)-(punisher_prop_belief*constants.c))
                        
                           
                        
                        
                        if interaction_idx == which_interaction_is_important:
                            is_important = True
                        else:
                            is_important = False
                        
                        if not is_important:
                            for pidx,p in enumerate(player_list.values()):
                                if punishers[pidx]:
                                    p.signalling_cost += -constants.c
                        
                        '''
                        for p in player_list.values():
                            if not is_important:
                                p.silly_game_utility += p.calc_exp_util(p.punisher_prop_belief,constants.d,constants.c,0)                       
                        ''' 
                            
                    individuals_left = len(player_list)/constants.players_per_group
                    #time_idx += int(1/(1-constants.d))*(1-constants.d)
                    time_idx = time_period
                    time_period_results[time_idx]=individuals_left
                    
                    
                print('d:',d_idx,d_sample,'group-id:',group_id,'left:'+str(individuals_left)+' @'+str(time_idx),'mean_util:'+str(np.mean([x.utility for x in player_list.values()])))
                if d_idx not in results:
                    results[d_idx] = dict()
                for time_per in time_period_results.keys():
                    if time_per not in results[d_idx]:
                        results[d_idx][time_per] = [time_period_results[time_per]]
                    else:
                        results[d_idx][time_per].append(time_period_results[time_per])
                
        for d_idx,d_sample in enumerate(all_d_samples):
            _X = list(sorted(results[d_idx].keys()))
            _Y = [sum([True if _p >= 0.02 else False for _p in results[d_idx][x]])/len(results[d_idx][x]) for x in _X]
            #_Y = [np.mean(results[d_idx][x]) for x in _X]
            #indxs = [idx for idx,x in enumerate(_Y) if x > 0.99]
            #max_xlim = max([max_xlim,max([x for idx,x in enumerate(_X) if idx in indxs])])
            
            ax.plot(_X,_Y,color=colors[d_idx],linestyle='-',marker='o',linewidth=0.5)
            
        ax.set_xlim([0, max_time_steps])
        plt.colorbar(sm)
        plt.show()
            
    def run_sim(self):
        fig, ax = plt.subplots(1, 1)
        
        constants.c = 0.02
        cmap = cm.coolwarm
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
        #all_d_samples = np.linspace(0,0.95,10)
        all_d_samples = [0,0.5,0.9]
        colors = cm.coolwarm(np.linspace(1,0,len(all_d_samples)))
        results = dict()
        for d_idx,d_sample in enumerate(all_d_samples):
            
            for group_id in np.arange(100):
                constants.d = d_sample
                imp_games_played,time_idx = 0,0
                player_list = {i:Player(i) for i in np.arange(constants.players_per_group)}
                ''' there should be constants.punisher_prop*len(player_list) number of punishers'''
                #punishers = np.random.choice([True,False],size=len(player_list),p=[constants.punisher_prop,1-constants.punisher_prop])
                time_period_results = {0:1}
                
                for time_period in np.arange(100):
                    
                    #print(time_period,'players:'+str(len(player_list)),'utils:'+str(np.mean([x.utility for x in player_list.values()]))+'-'+str(np.std([x.utility for x in player_list.values()])))
                    which_interaction_is_important = np.random.choice(np.arange(int(1/(1-constants.d))))
                    for interaction_idx in np.arange(int(1/(1-constants.d))):
                        for player_id in list(player_list.keys()):
                            if player_list[player_id].utility < 0:
                                del player_list[player_id]
                        if len(player_list) < 2:
                            continue
                        
                        if interaction_idx == which_interaction_is_important:
                            is_important = True
                        else:
                            is_important = False
                        
                        choice_list = list(player_list.keys())
                        all_games = []
                        
                        while len(choice_list) > 1:
                            player_ids = np.random.choice(choice_list,size=2,replace=False) if len(choice_list) > 2 else choice_list 
                            player_1 = player_list[player_ids[0]]
                            player_2 = player_list[player_ids[1]]
                            all_games.append(Game(player_1,player_2,None,None))
                            choice_list = [x for x in choice_list if x not in player_ids]
                        punisher_indexes = np.random.choice(list(player_list.keys()),size=int(constants.punisher_prop*len(player_list)),replace=False)
                        for p in player_list.keys():
                            if p in punisher_indexes:
                                player_list[p].is_punisher = True
                            else:
                                player_list[p].is_punisher = False
                        
                        for gidx,g in enumerate(all_games):
                            if is_important:
                                g.is_important = True
                            else:
                                g.is_important = False
                            
                            if np.random.random_sample() > 0.5:
                                g.player_1.assign_role(VictimRole())
                                g.player_2.assign_role(BystanderRole())
                            else:
                                g.player_2.assign_role(VictimRole())
                                g.player_1.assign_role(BystanderRole())
                                
                            
                        #if sum(is_important) > 1:
                        #    print('more than one imp game played',sum(is_important),len(all_games))
                        #imp_games_played += sum(is_important)
                        env = Environment(list(player_list.values()))
                        env.set_all_games(all_games)
                        for g in all_games:
                            g.set_env_ref(env)
                        badct = 0
                        for idx,g in enumerate(all_games):
                            g.is_important = True if is_important else False
                            g.player_1.playGame()  
                            g.player_2.playGame()  
                            '''
                            if g.player_1.utility < 0:
                                print('player_1 dropped',g.is_important,g.player_1.is_punisher,time_period,interaction_idx)
                            if g.player_2.utility < 0:
                                print('player_2 dropped',g.is_important,g.player_2.is_punisher,time_period,interaction_idx)
                            '''
                            
                    #print([x.utility for x in player_list.values()])        
                    individuals_left = sum([True if x.utility > 0 else False for x in player_list.values()])/constants.players_per_group
                    #time_idx += int(1/(1-constants.d))*(1-constants.d)
                    time_idx = time_period
                    time_period_results[time_idx]=individuals_left
                
                print('d:',d_idx,d_sample,'group-id:',group_id,'left:'+str(individuals_left)+' @'+str(time_idx),'mean_util:'+str(np.mean([x.utility for x in player_list.values()])))
                if d_idx not in results:
                    results[d_idx] = dict()
                for time_per in time_period_results.keys():
                    if time_per not in results[d_idx]:
                        results[d_idx][time_per] = [time_period_results[time_per]]
                    else:
                        results[d_idx][time_per].append(time_period_results[time_per])
        max_xlim = 90
        for d_idx,d_sample in enumerate(all_d_samples):
            _X = list(sorted(results[d_idx].keys()))
            _Y = [sum([True if _p >= 0.02 else False for _p in results[d_idx][x]])/len(results[d_idx][x]) for x in _X]
            #_Y = [np.mean(results[d_idx][x]) for x in _X]
            #indxs = [idx for idx,x in enumerate(_Y) if x > 0.99]
            #max_xlim = max([max_xlim,max([x for idx,x in enumerate(_X) if idx in indxs])])
            
            ax.plot(_X,_Y,color=colors[d_idx],linestyle='-')
            
        ax.set_xlim([0, max_xlim+10])
        plt.colorbar(sm)
        plt.show()
    
    

class Simulation:
    
    def run_sim(self):
        fig, ax = plt.subplots(1, 1)
        
        
        cmap = cm.coolwarm
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
        #all_d_samples = np.linspace(0,0.95,10)
        all_d_samples = [0]
        colors = cm.coolwarm(np.linspace(0,1,len(all_d_samples)))
        results = dict()
        for d_idx,d_sample in enumerate(all_d_samples):
            
            for group_id in np.arange(10):
                constants.d = d_sample
                imp_games_played,time_idx = 0,0
                player_list = {i:Player(i) for i in np.arange(constants.players_per_group)}
                ''' there should be constants.punisher_prop*len(player_list) number of punishers'''
                #punishers = np.random.choice([True,False],size=len(player_list),p=[constants.punisher_prop,1-constants.punisher_prop])
                time_period_results = {0:1}
                
                for time_period in np.arange(1,100,1):
                    show_bel_id = np.random.choice(list(player_list.keys()))
                    utils.plot_beta(player_list[show_bel_id].belief_alpha, player_list[show_bel_id].belief_beta)
                    #print(time_period,'players:'+str(len(player_list)),'utils:'+str(np.mean([x.utility for x in player_list.values()]))+'-'+str(np.std([x.utility for x in player_list.values()])))
                    which_interaction_is_important = np.random.choice(np.arange(int(1/(1-constants.d))))
                    for interaction_idx in np.arange(int(1/(1-constants.d))):
                        for player_id in list(player_list.keys()):
                            if player_list[player_id].utility < 0:
                                del player_list[player_id]
                        if len(player_list) < 2:
                            continue
                        
                        if interaction_idx == which_interaction_is_important:
                            is_important = True
                        else:
                            is_important = False
                        
                        choice_list = list(player_list.keys())
                        all_games = []
                        
                        while len(choice_list) > 1:
                            player_ids = np.random.choice(choice_list,size=2,replace=False) if len(choice_list) > 2 else choice_list 
                            player_1 = player_list[player_ids[0]]
                            player_2 = player_list[player_ids[1]]
                            all_games.append(Game(player_1,player_2,None,None))
                            choice_list = [x for x in choice_list if x not in player_ids]
                        punisher_indexes = np.random.choice(list(player_list.keys()),size=int(constants.punisher_prop*len(player_list)),replace=False)
                        for p in player_list.keys():
                            if p in punisher_indexes:
                                player_list[p].is_punisher = True
                            else:
                                player_list[p].is_punisher = False
                        
                        for gidx,g in enumerate(all_games):
                            if is_important:
                                g.is_important = True
                            else:
                                g.is_important = False
                            
                            if np.random.random_sample() > 0.5:
                                g.player_1.assign_role(VictimRole())
                                g.player_2.assign_role(BystanderRole())
                            else:
                                g.player_2.assign_role(VictimRole())
                                g.player_1.assign_role(BystanderRole())
                                
                            
                        #if sum(is_important) > 1:
                        #    print('more than one imp game played',sum(is_important),len(all_games))
                        #imp_games_played += sum(is_important)
                        env = Environment(player_list)
                        env.set_all_games(all_games)
                        for g in all_games:
                            g.set_env_ref(env)
                        badct = 0
                        for p in player_list.values():
                            if p.is_punisher:
                                p.utility = p.utility - constants.c
                        for idx,g in enumerate(all_games):
                            g.is_important = True if is_important else False
                            g.player_1.playGame()  
                            g.player_2.playGame()  
                            '''
                            if g.player_1.utility < 0:
                                print('player_1 dropped',g.is_important,g.player_1.is_punisher,time_period,interaction_idx)
                            if g.player_2.utility < 0:
                                print('player_2 dropped',g.is_important,g.player_2.is_punisher,time_period,interaction_idx)
                            '''
                        #plt.hist([x.utility for x in player_list.values()],density=True,bins=5)
                        #plt.title(str(time_period)+'-'+str(interaction_idx))
                        #plt.show()       
                    #print([x.utility for x in player_list.values()])        
                    individuals_left = sum([True if x.utility > 0 else False for x in player_list.values()])/constants.players_per_group
                    #time_idx += int(1/(1-constants.d))*(1-constants.d)
                    time_idx = time_period
                    time_period_results[time_idx]=individuals_left
                
                print('d:',d_idx,d_sample,'group-id:',group_id,'left:'+str(individuals_left)+' @'+str(time_idx),'mean_util:'+str(np.mean([x.utility for x in player_list.values()])))
                if d_idx not in results:
                    results[d_idx] = dict()
                for time_per in time_period_results.keys():
                    if time_per not in results[d_idx]:
                        results[d_idx][time_per] = [time_period_results[time_per]]
                    else:
                        results[d_idx][time_per].append(time_period_results[time_per])
        max_xlim = 90
        for d_idx,d_sample in enumerate(all_d_samples):
            _X = list(sorted(results[d_idx].keys()))
            _Y = [sum([True if _p >= 0.02 else False for _p in results[d_idx][x]])/len(results[d_idx][x]) for x in _X]
            #_Y = [np.mean(results[d_idx][x]) for x in _X]
            #indxs = [idx for idx,x in enumerate(_Y) if x > 0.99]
            #max_xlim = max([max_xlim,max([x for idx,x in enumerate(_X) if idx in indxs])])
            
            ax.plot(_X,_Y,color=colors[d_idx],linestyle='-')
            
        ax.set_xlim([0, max_xlim+10])
        plt.colorbar(sm)
        plt.show()

s = Simulation()
#constants.alpha = 30
#constants.beta = 20
'''baseline (0.02)'''
#constants.c = 0.09
'''baseline (0.01)'''
#constants.c = 0.07
'''baseline (0.005)'''
#constants.c = .035
'''baseline (0.002)'''
#constants.c = .0138
constants.c = 0.0035
constants.alpha = 1.2
constants.beta = 0.8
s.run_sim()