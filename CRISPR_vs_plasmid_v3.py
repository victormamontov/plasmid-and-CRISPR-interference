# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 08:41:54 2017

@author: marti
"""


import random as rnd
import numpy as np
import scipy as sp
import scipy.special
import matplotlib.pyplot as plt
from matplotlib import rc
import time
import os
from scipy.integrate import odeint

plt.rc('text', usetex=True)

                    
GEN_COLORS = ['#000068', '#0000bc', '#0000FF', '#0065ff', '#0094ff', '#00d4ff'] #new blue
SECONDARY_COLORS = ['#663201', '#a35101','#ef7600','#FD9632', '#fdb532', '#fdd732'] #new red


class CellCourse(object):
    def __init__(self, initial_value, total_rate, dup_prob):
        self.plasmid = [initial_value]
        self.timesteps = [0]
        self.plasmid_current = initial_value
        self.total_rate = total_rate
        self.dup_prob = dup_prob
        self.died = False
        pass
    
    def gillespie(self, sim_end):
        time = 0
        while True:
            step, new_number = self.gill_step()
            time = time + step
            if time < sim_end:
                self.timesteps.append(time)
                self.plasmid.append(new_number)
                self.plasmid_current = new_number
                if new_number == 0:
                    self.died = True
                    break
            else:
                break
    
    def gill_step(self):
        next_step = rnd.expovariate(self.total_rate[self.plasmid_current])
        if rnd.random() < self.dup_prob[self.plasmid_current]:
            next_number = self.plasmid_current + 1
        else:
            next_number = self.plasmid_current - 1
        return next_step, next_number
    
    def time_slice(self, slice_time):
        i = 0
        while self.timesteps[i] < slice_time:
            current = self.plasmid[i]
            i = i + 1
        return current
        
    
    
class DataCollection(object):
    def __init__(self, max_plasmid, d_rate, r_rate, chi, crispr_number, cell_d_rate, skew = 1):
        self.max_plasmid = int(max_plasmid)
        self.d_rate = float(d_rate)
        self.r_rate = float(r_rate)
        self.chi = float(chi)
        self.crispr_number = int(crispr_number)
        self.cell_d_rate = float(cell_d_rate)
        self.skew = float(skew)
        if self.is_data():
            file_name = self.make_name()
            self.read(file_name)
            self.datapoints = self.data.sum(1)[-1]
        else:
            self.data = np.zeros([max_plasmid+1, max_plasmid+1], dtype='int')
            self.datapoints = 0
        pass
    
    def make_name(self):
        if self.skew == 1:
            file_name = "%s_%s_%s_%s_%s_%s.txt" %(self.max_plasmid, self.d_rate, 
                                   self.r_rate, self.chi, self.crispr_number, self.cell_d_rate)
        else:
            file_name = "%s_%s_%s_%s_%s_%s_%s.txt" %(self.max_plasmid, self.d_rate, 
                                   self.r_rate, self.chi, self.crispr_number, self.cell_d_rate, self.skew)
        return file_name
        
    def is_data(self):
        """Tests if there is a file with data that correspond to current constants"""
        file_name = self.make_name()
        if os.path.isfile(file_name):
            return True
        else:
            return False
        
    def get_rate_prob(self):
        """Returns total rate and probability of plasmid duplication"""
        plasmid = np.array(range(0,self.max_plasmid+1))
        dup_rate = self.d_rate*self.max_plasmid*(plasmid/self.max_plasmid)**(1/self.skew)*(1
                                                 -(plasmid/self.max_plasmid)**(1/self.skew))
        cut_rate = self.r_rate*(self.crispr_number + self.chi + plasmid
                                - np.sqrt((self.crispr_number + self.chi + plasmid)**2 
                                          - 4*self.crispr_number*plasmid))/2
        total_rate = dup_rate + cut_rate
        dup_prob = dup_rate/total_rate
        return (total_rate, dup_prob, dup_rate, cut_rate)
    
    def generate(self, repetitions):
        total_rate, dup_prob, dup_rate, cut_rate = self.get_rate_prob()
        for j in range(repetitions):
            for i in range(1,self.max_plasmid+1):
                new_cell = CellCourse(i, total_rate, dup_prob)
                new_cell.gillespie(1/self.cell_d_rate)
                end_val = new_cell.plasmid_current
                self.data[i, end_val] += 1
            if j%1000 == 0:
                print("Repetitions: " + str(j))
        self.datapoints = self.datapoints + repetitions

    def write(self):
        file_name = self.make_name()
        np.savetxt(file_name, self.data, fmt = '%i')
        
    def read(self, file_name):
        name = file_name.strip('.txt')
        params = name.split('_')
        if len(params) == 6:
            max_plasmid, d_rate, r_rate, chi, crispr_number, cell_d_rate = params
            skew = 1
        else:
            max_plasmid, d_rate, r_rate, chi, crispr_number, cell_d_rate, skew = params
        self.max_plasmid = int(max_plasmid)
        self.d_rate = float(d_rate)
        self.r_rate = float(r_rate)
        self.chi = float(chi)
        self.crispr_number = int(crispr_number)
        self.cell_d_rate = float(cell_d_rate)
        self.data = np.loadtxt(file_name, dtype='int')
        self.skew = float(skew)
        
    def count_eq(self, dup_rate, cut_rate):
        eq = None
        for i in range(len(dup_rate)):
            if dup_rate[i] > cut_rate[i]:
                eq = i
        return eq
    
    def count_unstable_eq(self, dup_rate, cut_rate):
        eq = None
        for i in range(len(dup_rate)):
            if cut_rate[i] > dup_rate[i]:
                eq = i
            if dup_rate[i] > cut_rate[i] and eq != None:
                return eq
        return None
        
    def plot_duplication_degradation(self, ylim = None, all_labels = True, wide = False, labelnumbers = False):
        total_rate, dup_prob, dup_rate, cut_rate = self.get_rate_prob()
        eq = self.count_eq(dup_rate, cut_rate)
        uneq = self.count_unstable_eq(dup_rate, cut_rate)
        fig = plt.figure()
        if wide:
            fig.set_size_inches(13, 7, forward=True)
        else:
            fig.set_size_inches(10, 8, forward=True)
        ax = fig.add_subplot(111)
        ax.plot(list(range(0,self.max_plasmid+1)), dup_rate, label = "Replication", linewidth=3.0, color = '#0000FF')
        ax.plot(list(range(0,self.max_plasmid+1)), cut_rate, label = "Interference", linewidth=3.0, color = '#FD9632')
        ax.legend(prop={'size': 26})
        plt.tick_params(labelsize=26)
        plt.gcf().subplots_adjust(bottom=0.15) 
        ax.set_xlabel("Plasmid number", fontsize=26)
        ax.set_ylabel("Replication and interference rates", fontsize=26)
        if ylim == None:
            ax.set_ylim([0,max(dup_rate + cut_rate)*1.1])
        else:
            ax.set_ylim([0,ylim])
        if all_labels == False:
            ylabels = [item.get_text() for item in ax.get_yticklabels()]
            print(ylabels)
            y_string_labels = ['']*len(ylabels)
            ax.set_yticklabels(y_string_labels)
            xticks = ax.get_xticks()
            xticks = [x for x in xticks if 0<=x<=self.max_plasmid]
            if uneq != None:
                unpos = 0
                for i in range(len(xticks)):
                    if xticks[i] < uneq:
                        unpos = i
                unpos += 1
                xticks.insert(unpos, uneq)
            if eq != None:
                pos = 0
                for i in range(len(xticks)):
                    if xticks[i] < eq:
                        pos = i
                pos += 1
                xticks.insert(pos, eq)
            ax.set_xticks(xticks)
            x_string_labels = ['']*len(xticks)
            x_string_labels[0] = '0'
            if labelnumbers:
                if eq != None:
                    x_string_labels[pos] = r'$\mathrm{[Pl]_{eq}}$' + ' = ' + str(eq)
                if uneq != None:
                    x_string_labels[unpos] = r'$\mathrm{[Pl]_{bif}}$' + ' = ' + str(uneq)
                x_string_labels[-1] = r'$\mathrm{[Pl]_{st}}$' + ' = ' + str(self.max_plasmid)
            else:
                if eq != None:
                    x_string_labels[pos] = r'$\mathrm{[Pl]_{eq}}$'
                if uneq != None:
                    x_string_labels[unpos] = r'$\mathrm{[Pl]_{bif}}$'
                x_string_labels[-1] = r'$\mathrm{[Pl]_{st}}$'
            print(x_string_labels)
            ax.set_xticklabels(x_string_labels)
        return fig
    
    
class StatsAssembly(object):
    def __init__(self, max_plasmid, d_rate, r_rate, chi, crispr_number, cell_d_rate, skew = 1, simulate = True):
        self.data = DataCollection(max_plasmid, d_rate, r_rate, chi, crispr_number, cell_d_rate, skew)
        if simulate == True:
            self.simulate()
        pass
    
    def simulate(self):
        if self.data.datapoints < 15000:
            self.data.generate(5000)
        elif self.data.datapoints > 30000:
            pass
        else:
            pass
        self.data.write()
        pass
        
    def get_growth_matrix(self):
        count_matrix = self.data.data
        total_counts = count_matrix.sum(axis=1)
        matrix_size = len(count_matrix)
        prob_matrix = np.zeros([matrix_size, matrix_size])
        for i in range(matrix_size):
            if total_counts[i] == 0:
                prob_matrix[i] = 0
            else:
                prob_matrix[i] = count_matrix[i]/total_counts[i]
        prob_matrix[0,0] = 1
        return prob_matrix
        
    def get_duplication_matrix(self):
        matrix_size = len(self.data.data)
        prob_matrix = np.zeros([matrix_size, matrix_size])
        for i in range(matrix_size):
            for j in range(matrix_size):
                prob_matrix[i,j] = scipy.special.comb(i,j)*0.5**i
        return prob_matrix
    
    def setup_population(self, plasmid_number):
        population_stats = []
        population = np.zeros([len(self.data.data)])
        population[plasmid_number] = 1
        return population_stats, population       
    
    def simulation_assembly_probabilistic(self, plasmid_number, generations):
        population_stats, population = self.setup_population(plasmid_number)
        growth_matrix = self.get_growth_matrix()
        growth_matrix = self.discontinuos_smoothing(growth_matrix) #!!!!!!!!!!!!!!!
        duplication_matrix = self.get_duplication_matrix()
        population_stats.append(population)
        for i in range(generations):
            population = np.dot(population,growth_matrix)
            population_stats.append(population)
            population = np.dot(population, duplication_matrix)
        return population_stats
    
    def simulation_assembly_master(self, plasmid_number, generations):
        population_stats, population = self.setup_population(plasmid_number)
        _,__, dup, deg = self.data.get_rate_prob()
        population_stats.append(population)
        t = np.linspace(0, 1/self.data.cell_d_rate, 10)
        duplication_matrix = self.get_duplication_matrix()
        for i in range(generations):
            if i%20 == 0:
                print('Generation: %s' %(i+1))
            y = odeint(master, population, t, args = (dup, deg))
            population = y[-1]
            population_stats.append(population)
            population = np.dot(population, duplication_matrix)
        return population_stats
    
    def adjust_stats(self, stats):
        new_stats = stats[:]
        for i, stat in enumerate(new_stats):
            new_stats[i] = stat/(1-stat[0])
            new_stats[i][0] = 0
        return new_stats
   
   
    def plot_generations(self, stats, generations = None, all_labels = True, 
                         colorset = None, zeros_text = True, asymptotic = False, 
                         figure = None, markerstyle = None, legend = True,
                         labelnumbers = False):
        if colorset == None:
            colorset = GEN_COLORS
        for i, stat in enumerate(stats[0]):
            stats[0][i] += 10**-40
        if isinstance(generations, int):
            generations = [generations]
        if generations == None:
            generations = list(range(len(stats)))
        if figure == None:
            gen_fig = plt.figure()
            gen_fig.set_size_inches(13, 7, forward=True)
            gen_ax = gen_fig.add_subplot(111)
        else:
            gen_fig = figure
            gen_ax = gen_fig.gca()
        pl_num_array = list(range(len(stats[0])))
        if markerstyle == None:
            gen0marker = '^'
            allmarker = 'o'
        else:
            gen0marker = markerstyle
            allmarker = markerstyle
        for i, gen in enumerate(generations):
            if gen == 0:
                gen_ax.plot(pl_num_array[1:],stats[gen][1:], marker=gen0marker, markersize = 10.0, linestyle = 'None', 
                            linewidth=3.0, label = "Generation: " + str(gen), color = colorset[i])               
            else:
                gen_ax.plot(pl_num_array[1:],stats[gen][1:], marker=allmarker, markersize = 5.0, linestyle = 'None', 
                            linewidth=3.0, label = "Generation: " + str(gen), color = colorset[i])
                gen_ax.plot(-0.3,stats[gen][0], marker=allmarker, markersize = 8.0, linestyle = 'None', 
                            linewidth=3.0, label = None, color = colorset[i], mfc = 'none', markeredgewidth=3)
        if asymptotic:
            gen_ax.plot(pl_num_array[1:], stats[-1][1:], color = 'k', linewidth=2.0, label = 'Asymptotic \n distribution')
        if legend:
            gen_ax.legend(prop={'size': 19})
        plt.tick_params(labelsize=26)
        plt.gcf().subplots_adjust(bottom=0.15)          
        gen_fig.show()
        ymin=10**(-11)
        ymax=1.5
        gen_ax.set_ylim([ymin, ymax])
        gen_ax.set_yscale('log')
        yticks = []
        for i in range(-11,0,2):
            yticks.append(10**i)
        yticks.append(1)
        gen_ax.set_yticks(yticks)
        gen_ax.set_xlabel("Plasmid number", fontsize=26)
        gen_ax.set_ylabel("Probability", fontsize=26)
        if zeros_text:
            if 1-stats[-1][0] < 0.1:
                gen_ax.text(10,0.4,"Fraction of cells with plasmid: " + "%.1e" %(1-stats[-1][0]), fontsize=20)
            else:
                gen_ax.text(10,0.4,"Fraction of cells with plasmid: " + "%.1f" %(1-stats[-1][0]), fontsize=20)
        total_rate, dup_prob, dup_rate, cut_rate = self.data.get_rate_prob()
        eq = self.data.count_eq(dup_rate, cut_rate)
        uneq = self.data.count_unstable_eq(dup_rate, cut_rate)
        if all_labels == False:
            xticks = gen_ax.get_xticks()
            xticks = [x for x in xticks if 0<=x<=self.data.max_plasmid]
            if uneq != None:
                unpos = 0
                for i in range(len(xticks)):
                    if xticks[i] < uneq:
                        unpos = i
                unpos += 1
                xticks.insert(unpos, uneq)
            if eq != None:
                pos = 0
                for i in range(len(xticks)):
                    if xticks[i] < eq:
                        pos = i
                pos += 1
                xticks.insert(pos, eq)
            gen_ax.set_xticks(xticks)
            x_string_labels = ['']*len(xticks)
            x_string_labels[0] = '0'
            if labelnumbers:
                if eq != None:
                    x_string_labels[pos] = r'$\mathrm{[Pl]_{eq}}$' + ' = ' + str(eq)
                if uneq != None:
                    x_string_labels[unpos] = r'$\mathrm{[Pl]_{bif}}$'  + ' = ' + str(uneq)
                x_string_labels[-1] = r'$\mathrm{[Pl]_{st}}$' + ' = ' + str(self.data.max_plasmid)
            else:
                if eq != None:
                    x_string_labels[pos] = r'$\mathrm{[Pl]_{eq}}$'
                if uneq != None:
                    x_string_labels[unpos] = r'$\mathrm{[Pl]_{bif}}$'
                x_string_labels[-1] = r'$\mathrm{[Pl]_{st}}$'
            print(x_string_labels)
            gen_ax.set_xticklabels(x_string_labels)
        return gen_fig
    
    def plot_weight_matrix(self):
        data = self.get_growth_matrix()
        max_val = data[10:][10:].max()
        print(max_val)
        fig = plt.figure()
        fig.set_size_inches(8, 8, forward=True)
        ax = fig.add_subplot(111)
        ax.contourf(data, np.linspace(0,max_val, 100))
        ax.set_xlabel("Plasmid number after cell growth")
        ax.set_ylabel("Initial plasmid number")
        return fig
    
    def plot_weight_matrix_cleaned(self):
        data = self.get_growth_matrix()
        data = self.discontinuos_smoothing(data)
        max_val = data[3:][3:].max()
        fig = plt.figure()
        fig.set_size_inches(8, 8, forward=True)
        ax = fig.add_subplot(111)
        ax.contourf(data, np.linspace(0,max_val, 100))
        ax.set_xlabel("Plasmid number after cell growth")
        ax.set_ylabel("Initial plasmid number")
        return fig
        
    def plot_next_generation_matrix(self):
        fig = plt.figure()
        fig.set_size_inches(8, 8, forward=True)
        ax = fig.add_subplot(111)
        growth_matrix = self.get_growth_matrix()
        duplication_matrix = self.get_duplication_matrix()
        ax.contourf(np.dot(growth_matrix, duplication_matrix), np.linspace(0,0.3, 100))
        ax.set_xlabel("Next generation plasmid number")
        ax.set_ylabel("Initial plasmid number")
        return fig
    
    def discontinuos_smoothing(self, old_matrix):
        matrix = old_matrix.copy()
        shape = matrix.shape
        for i in range(1, shape[1]):
            max_id = matrix[i][1:].argmax()
            j = max_id
            while matrix[i][j] != 0 and j != shape[1]-1:
                j = j + 1
            end_id = j
            j = max_id
            while matrix[i][j] != 0 and j != 0:
                j = j - 1
            beginning_id = j
            for j in range(end_id, shape[1]):
                if matrix[i][j] != 0:
                    temp = matrix[i][j]
                    matrix[i][j] = 0
                    for k in range(end_id,j+1):
                        matrix[i][k] += temp/(j - end_id + 1)
            for j in reversed(range(1,beginning_id+1)):
                if matrix[i][j] != 0:
                    temp = matrix[i][j]
                    matrix[i][j] = 0
                    for k in range(j, beginning_id+1):
                        matrix[i][k] += temp/(beginning_id - j + 1)
        return matrix

def reseed_random(population, reseeds_number, generations, param):
    new_pop_list = []
    max_plasmid = len(population)
    for i in range(0, reseeds_number):
        init_number = np.random.choice(max_plasmid, p = population)
        sa = StatsAssembly(*param, simulate = False)
        stats = sa.simulation_assembly_master(init_number, generations)
        new_pop_list.append(stats[-1])
    return new_pop_list

def reseed_random_nonzero(population, reseeds_number, generations, param):
    new_pop_list = []
    max_plasmid = len(population)
    correction = sum(population[1:])
    print(sum(population))
    population = [x/correction for x in population]
    population[0] = 0
    for i in range(0, reseeds_number):
        init_number = np.random.choice(max_plasmid, p = population)
        print(init_number)
        sa = StatsAssembly(*param, simulate = False)
        stats = sa.simulation_assembly_master(init_number, generations)
        new_pop_list.append(stats[-1])
    return new_pop_list

def master(y, t, inc, dec):
    pos = len(y)-1
    dy = np.zeros([pos+1])
    dy[0] = dec[1]*y[1] - inc[0]*y[0]
    for i in range(1, pos):
        dy[i] = inc[i-1]*y[i-1] + dec[i+1]*y[i+1] - (inc[i] + dec[i])*y[i]
    dy[pos] = inc[pos-1]*y[pos-1] - dec[pos]*y[pos]
    return dy
    
def save_figure(fig, parameters, key, forward = False, form = 'pdf'):
    file_name = "figures/"
    if forward:
        file_name = file_name + key + '_'
        for param in parameters:
            file_name = file_name + "%s" %(param)
        file_name = file_name + "." + form
    else:
        for param in parameters:
            file_name = file_name + "%s" %(param)
        file_name = file_name + '_' + key + "." + form
    fig.savefig(file_name, dpi = 600, format=form, transparent = True)
    
def fig1():
    par1a = [100, 0.3, 0.5, 1, 18, 0.05]    #fig 1A
    main = StatsAssembly(*par1a, simulate = False)
    fig1a = main.data.plot_duplication_degradation(ylim = 9, all_labels=False)
    save_figure(fig1a,par1a,"1A", forward = True)
    par1b = [100, 0.3, 0.5, 1, 10, 0.05]    #fig 1B
    main = StatsAssembly(*par1b, simulate = False)
    fig1b = main.data.plot_duplication_degradation(ylim = 9, all_labels=False)
    save_figure(fig1b,par1b,"1B", forward = True)    
    par1c = [100, 0.3, 0.2, 1, 20, 0.05]    #fig 1C
    main = StatsAssembly(*par1c, simulate = False)
    fig1c = main.data.plot_duplication_degradation(ylim = 9, all_labels=False)
    save_figure(fig1c,par1c,"1C", forward = True)

def fig2():
    param = [100, 0.3, 0.5, 1, 10, 0.05] #fig 2 = fig1B
    main = StatsAssembly(*param, simulate = False)
    fig2a = main.data.plot_duplication_degradation(ylim = 9, all_labels=False, wide=True, labelnumbers=True)
    save_figure(fig2a,param,"2A", forward = True)
    stats = main.simulation_assembly_master(1,30)
    fig2b = main.plot_generations(stats,[0,1,2,5,10,20], all_labels=False, labelnumbers=True)
    save_figure(fig2b,param,"2B", forward = True)
    stats = main.simulation_assembly_master(100,30)
    fig2c = main.plot_generations(stats,[0,1,2,5,10,20], all_labels=False, labelnumbers=True)
    save_figure(fig2c,param,"2C", forward = True)
    pass

def figM1():    
    param = [100, 0.3, 0.5, 1, 10, 0.05] #fig 2 = fig1B
    main = StatsAssembly(*param, simulate = False)
    stats = main.simulation_assembly_master(1,200)
    stats = main.adjust_stats(stats)
    figM1a = main.plot_generations(stats,[0,1,2,3,4,5], all_labels=False, zeros_text=False, markerstyle='^', legend = False, labelnumbers=True)
    stats = main.simulation_assembly_master(100,200)
    stats = main.adjust_stats(stats)
    figM1a = main.plot_generations(stats,[0,1,2,3,4,5], all_labels=False, zeros_text=False, 
                                   asymptotic=True, figure = figM1a, colorset=SECONDARY_COLORS, markerstyle='^',
                                   legend = False, labelnumbers=True)
    save_figure(figM1a,param,"M1", forward = True)
    

print("starting collction of DB")
start = time.time()


fig1()
fig2()
figM1()




    
