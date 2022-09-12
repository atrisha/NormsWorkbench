'''
Created on 19 Aug 2022

@author: Atrisha
'''
import constants
import scipy.stats as spstat
import math
import numpy as np

class Environment:
    
    def __init__(self, player_list):
        self.player_list = player_list
        self.num_punishers = sum([x.is_punisher for x in player_list.values()])
        self.num_non_punishers = len(player_list) - self.num_punishers
        
    def set_all_games(self,all_games):
        self.all_games = all_games

class Game:
    
    def __init__(self,player_1,player_2,env_ref,is_important):
        self.player_1 = player_1
        self.player_2 = player_2
        self.env_ref = env_ref
        self.player_1.set_current_game(self)
        self.player_2.set_current_game(self)
        self.is_important = is_important

    def get_other_player(self,this_player_id):
        return self.player_1 if self.player_2.id == this_player_id else self.player_2
    
    def get_player(self,this_player_id):
        return self.player_1 if self.player_1.id == this_player_id else self.player_2
    
    def set_env_ref(self,env_ref):
        self.env_ref = env_ref

class RoleInterface:
    
    def playRole(self,game: Game, player_id: int):
        ''' play the role that this type of game requires''' 
        pass

class Role(RoleInterface):
    
    def playRole(self, game:Game, player_id: int):
        ''' plays the role in the game and update '''   
        pass

class PlayerUtils:
    
    def calc_exp_util(self,theta,d,c,period_util):
        scaled_disc = 1- (1-d)*(1-constants.discount_factor)
        #scaled_disc = constants.discount_factor 
        sum = period_util
        disc_factor_multiplier = scaled_disc
        #exp_util =  (0.5/(1-constants.discount_factor))* (((2*theta-1)*constants.R) - (theta*c))
        exp_util = ((0.5/(1-constants.discount_factor))* (((2*theta-1)*constants.R) - (2*theta*c))) - ((0.5/(1-scaled_disc))*d*theta*c)
        #- ((0.5/(1-scaled_disc))*d*theta*c)
        #exp_util = - ((0.5/(1-scaled_disc))*constants.d*theta*c)
        #sum += exp_util
        return round(exp_util,5)

class Player(PlayerUtils):
    
    def __init__(self,id):
        self.id = id
        self.utility = 0
        self.belief_alpha = constants.alpha
        self.belief_beta = constants.beta
        self.is_punisher = None
        self.signalling_cost = []
    
    def bayes_update_punishers(self,obs_vals=None):
        if obs_vals is None:
            game = self.get_current_game()
            if self.belief_alpha is None:
                self.belief_alpha = constants.alpha
            if self.belief_beta is None:
                self.belief_beta = constants.beta 
            self.belief_alpha += game.env_ref.num_punishers
            self.belief_beta += game.env_ref.num_non_punishers
        else:
            if self.belief_alpha is None:
                self.belief_alpha = constants.alpha
            if self.belief_beta is None:
                self.belief_beta = constants.beta 
            self.belief_alpha += obs_vals[0]
            self.belief_beta += obs_vals[1]
            
    def assign_role(self, role: Role):
        self.role = role
    
    def opt_in_check(self):
        if self.utility < 0:
            return False
        else:
            return True
    
    def playGame(self):
        game = self.get_current_game()
        self.bayes_update_punishers()
        punishers_observed = self.role.playRole(game,self)
        self.bayes_update_punishers(punishers_observed)
        punisher_prop_sample = self.sample_punisher_basedon_belief()
        #util_val = constants.calc_sum_util(game_util, constants.d)
        #util_val = constants.calc_update_util(game_util,(self.belief_alpha)/(self.belief_alpha+self.belief_beta), constants.d, constants.c)
        #self.util_val_imp_games = self.calc_exp_util(punisher_prop_sample, constants.d, constants.c, 0)
        #punisher_cost = 0 if not self.is_punisher else -constants.c
        #self.signalling_cost.append(punisher_cost)
        util_val = self.calc_exp_util(punisher_prop_sample, constants.d, constants.c, 0)
        #self.utility = self.util_val_imp_games - sum(self.signalling_cost)
        self.utility += util_val
        
        
    def set_current_game(self,game):
        self.game = game
        
    def get_current_game(self):
        return self.game
    
    def sample_punisher_basedon_belief(self):
        self.punisher_prop_belief = np.random.beta(self.belief_alpha,self.belief_beta)
        return self.punisher_prop_belief

 
class VictimRole(Role):
    
    def playRole(self, game:Game, player: Player):
        ''' plays the Victim role in the game and update '''   
        other_player =  game.get_other_player(player.id)
        punishers_observed = 0
        if player.is_punisher:
            punishers_observed += 1
        if other_player.is_punisher:
            punishers_observed += 1
        non_punishers_observed = 2-punishers_observed
        return (punishers_observed,non_punishers_observed)
        '''
        punisher_cost = constants.c if player.is_punisher else 0
        if game.is_important:
            if isinstance(player.role, VictimRole) and other_player.is_punisher:
                return constants.R - punisher_cost
            elif isinstance(player.role, VictimRole) and not other_player.is_punisher:
                return -constants.R - punisher_cost
            else:
                return 0
        else:
            return 0
        '''

class BystanderRole(Role):
    
    def playRole(self, game:Game, player):
        ''' plays the Bystander role in the game and update '''   
        if player.is_punisher:
            return (1,0)
        else:
            return (0,1)
        '''
        if player.is_punisher:
            return -constants.c
        else:
            return 0
        '''

        
        
        
        