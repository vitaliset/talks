import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio
from scipy.stats import beta

class politica_aleatoria(object):
    
    def __init__(self, lista_bandidos):
        
        self.nome = 'politica_aleatoria'
        self.lista_bandidos = lista_bandidos
        self.numero_de_bandidos = len(lista_bandidos)
        self.treinado = False
        self.melhor_bandido = np.argmax(np.asarray(lista_bandidos))
        
    def train(self, numero_jogadas_experimentais):
        
        
        self.tempo_total_treinamento = numero_jogadas_experimentais
        # estimaremos o parâmetro da bernoulli associada a cada bandido usando o estimador de 
        # máxima verossimelhança da proporção olhando:
        # (quantas vezes ganhamos do bandido_k)/(quantas vezes jogamos contra o bandido_k)
        self.parametros_estimados = list(0.5*np.ones(self.numero_de_bandidos))
        self.vitoria_contra_o_bandido = np.zeros(self.numero_de_bandidos)
        self.jogadas_com_o_bandido = np.zeros(self.numero_de_bandidos)
        
        # numero de vitorias durante o treinamento.
        self.wins_exploration = 0
        # historico de qual indice foi escolhido em cada rodada do treinamento.
        self.historico_indice = []
        # historico do resultado em cada rodada do treinamento.
        self.historico_resultado = []
        
        # pra cada bandido, to guardando quando ele foi escolhido e qual foi o resultado daquela rodada.
        # usamos esse dicionário para criar os gráficos.
        self.historico = {str(indice):[[],[]] for indice in list(range(self.numero_de_bandidos))}
        # self.historico[bandido_k] = [[(indice da rodada)],[(resultado da rodada)]]
        
        # também guardaremos o histórico dos parâmetros dos bandidos estimados.
        self.historico_par = []
        
        # regret
        self.regret_historico = []
        regret_acumulado = 0
        
        for i in range(numero_jogadas_experimentais):
            lista_auxiliar = []

            # definindo quem vai jogar nessa rodada {i}.
            indice = np.random.randint(self.numero_de_bandidos)
            
            # jogando contra o bandido selecionado.
            resultado = np.random.binomial(1,self.lista_bandidos[indice])
            
            # guardando as informações históricas.
            self.historico_indice.append(indice)
            self.historico_resultado.append(resultado)
            self.wins_exploration += resultado
            
            self.historico[str(indice)][0].append(i)
            self.historico[str(indice)][1].append(resultado)
            
            self.vitoria_contra_o_bandido[indice] += resultado
            self.jogadas_com_o_bandido[indice] += 1
            
            for k in range(self.numero_de_bandidos):
                if not self.jogadas_com_o_bandido[k] == 0:
                    lista_auxiliar.append(self.vitoria_contra_o_bandido[k]/self.jogadas_com_o_bandido[k])
                else:
                    lista_auxiliar.append(0.5)
            
            regret_n = - self.lista_bandidos[indice] + self.melhor_bandido
            regret_acumulado += regret_n
            self.regret_historico.append(regret_acumulado)
            
            self.historico_par.append(lista_auxiliar)
            self.parametros_estimados = lista_auxiliar
            
        # terminado o treinamento. temos os estimadores do parâmetros.
        # escolhemos o indice do bandido com o melhor parâmetro.
        self.indice_best_bandido = np.argmax(self.parametros_estimados)
        self.treinado = True

    def play(self, numero_jogadas):
        
        self.wins_after = 0
        if self.treinado:
            for i in range(numero_jogadas):
                resultado = np.random.binomial(1,self.lista_bandidos[self.indice_best_bandido]) 
                self.wins_after += resultado
                
                
class politica_epsilon_greedy(object):
    
    def __init__(self, lista_bandidos):
        
        self.nome = 'politica_epsilon_greedy'
        self.lista_bandidos = lista_bandidos
        self.numero_de_bandidos = len(lista_bandidos)
        self.indice_best_bandido = 0
        self.treinado = False
        self.melhor_bandido = np.argmax(np.asarray(lista_bandidos))
        
    def train(self, numero_jogadas_experimentais, epsilon = 0.5, decreasing = False):
    
        if decreasing:
            self.nome = 'politica_epsilon_greedy_decreasing'
        
        self.tempo_total_treinamento = numero_jogadas_experimentais
        
        self.parametros_estimados = list(0.5*np.ones(self.numero_de_bandidos))
        self.vitoria_contra_o_bandido = np.zeros(self.numero_de_bandidos)
        self.jogadas_com_o_bandido = np.zeros(self.numero_de_bandidos)

        self.wins_exploration = 0
        self.historico_indice = []
        self.historico_resultado = []
 
        self.historico_epsilon = [[],[]]
    
        self.historico = {str(indice):[[],[]] for indice in list(range(self.numero_de_bandidos))}
        self.historico_par = []
        
        self.regret_historico = []
        regret_acumulado = 0
        
        for i in range(numero_jogadas_experimentais):
            lista_auxiliar = []
            eps = epsilon*(1 - int(decreasing)*i/numero_jogadas_experimentais)
            epsilon_resultado = (np.random.uniform() > (1 - eps))
            
            if i==0 or epsilon_resultado:
                indice = np.random.randint(self.numero_de_bandidos)
                resultado = np.random.binomial(1, self.lista_bandidos[indice]) 
                
                # guardando as informações históricas.
                self.historico_indice.append(indice)
                self.historico_resultado.append(resultado)
                self.wins_exploration += resultado

                self.historico[str(indice)][0].append(i)
                self.historico[str(indice)][1].append(resultado)
                
                self.historico_epsilon[0].append(0)
                self.historico_epsilon[1].append(eps)
                
                self.vitoria_contra_o_bandido[indice] += resultado
                self.jogadas_com_o_bandido[indice] += 1
                
                regret_n = - self.lista_bandidos[indice] + self.melhor_bandido
                regret_acumulado += regret_n
                self.regret_historico.append(regret_acumulado)
                
            else: 
                indice = self.indice_best_bandido
                resultado = np.random.binomial(1, self.lista_bandidos[indice])
                
                # guardando as informações históricas.
                self.historico_indice.append(indice)
                self.historico_resultado.append(resultado)
                self.wins_exploration += resultado

                self.historico[str(indice)][0].append(i)
                self.historico[str(indice)][1].append(resultado)
                
                self.historico_epsilon[0].append(1)
                self.historico_epsilon[1].append(eps)
                
                self.vitoria_contra_o_bandido[indice] += resultado
                self.jogadas_com_o_bandido[indice] += 1
                
                regret_n = - self.lista_bandidos[indice] + self.melhor_bandido
                regret_acumulado += regret_n
                self.regret_historico.append(regret_acumulado)

            for k in range(self.numero_de_bandidos):
                if not self.jogadas_com_o_bandido[k] == 0:
                    lista_auxiliar.append(self.vitoria_contra_o_bandido[k]/self.jogadas_com_o_bandido[k])
                else:
                    lista_auxiliar.append(0.5)
            
            self.historico_par.append(lista_auxiliar)
            self.parametros_estimados = lista_auxiliar
            self.indice_best_bandido = np.argmax(np.asarray(lista_auxiliar))
        
        self.treinado = True
        
        
class politica_thompson(object):
    
    def __init__(self, lista_bandidos):
        
        self.nome = 'politica_thompson'
        self.lista_bandidos = lista_bandidos
        self.numero_de_bandidos = len(lista_bandidos)
        self.indice_best_bandido = 0
        self.treinado = False
        self.melhor_bandido = np.argmax(np.asarray(lista_bandidos))
        
    def train(self, numero_jogadas_experimentais):
    
        self.tempo_total_treinamento = numero_jogadas_experimentais
        
        self.parametros_estimados = list(0.5*np.ones(self.numero_de_bandidos))
        self.vitoria_contra_o_bandido = np.zeros(self.numero_de_bandidos)
        self.derrota_contra_o_bandido = np.zeros(self.numero_de_bandidos)
        self.jogadas_com_o_bandido = np.zeros(self.numero_de_bandidos)

        self.wins_exploration = 0
        self.historico_indice = []
        self.historico_resultado = []
        
        self.historico = {str(indice):[[],[]] for indice in list(range(self.numero_de_bandidos))}
        self.historico_par = []
        
        self.historico_vit = []
        self.historico_der = []
        self.historico_estimado_da_rodada = []
        
        self.regret_historico = []
        regret_acumulado = 0
        
        for i in range(numero_jogadas_experimentais):
        
            estim_ = []
            for f in range(self.numero_de_bandidos):
                estim_.append(np.random.beta(self.vitoria_contra_o_bandido[f] + 1, self.derrota_contra_o_bandido[f] + 1))    

            indice = np.argmax(estim_)
            
            self.historico_estimado_da_rodada.append(estim_)

            resultado = np.random.binomial(1, self.lista_bandidos[indice]) 
            # guardando as informações históricas.
            self.historico_indice.append(indice)
            self.historico_resultado.append(resultado)
            self.wins_exploration += resultado
            
            self.historico[str(indice)][0].append(i)
            self.historico[str(indice)][1].append(resultado)
            
            self.vitoria_contra_o_bandido[indice] += resultado
            self.derrota_contra_o_bandido[indice] += 1 - resultado
            self.jogadas_com_o_bandido[indice] += 1

            lista_auxiliar = []
            lista_auxiliar2 = []
            lista_auxiliar3 = []
            
            for k in range(self.numero_de_bandidos):
                
                lista_auxiliar2.append(self.vitoria_contra_o_bandido[k])
                lista_auxiliar3.append(self.derrota_contra_o_bandido[k])
                
                if not self.jogadas_com_o_bandido[k] == 0:
                    lista_auxiliar.append(self.vitoria_contra_o_bandido[k]/self.jogadas_com_o_bandido[k])
                else:
                    lista_auxiliar.append(0.5)
            
            self.historico_par.append(lista_auxiliar)
            self.historico_vit.append(lista_auxiliar2)
            self.historico_der.append(lista_auxiliar3)
            
            self.parametros_estimados = lista_auxiliar
            
            regret_n = - self.lista_bandidos[indice] + self.melhor_bandido
            regret_acumulado += regret_n
            self.regret_historico.append(regret_acumulado)

        self.indice_best_bandido = np.argmax(self.parametros_estimados)
        self.treinado = True
        
    def play(self, numero_jogadas):
        
        self.wins_after = 0
        if self.treinado:
            for i in range(numero_jogadas):
                resultado = np.random.binomial(1,self.lista_bandidos[self.indice_best_bandido]) 
                self.wins_after += resultado
                
                
def gerando_gif(politics, lista_bandidos = [], epsilon_greedy = False, thompson = False, segundos_frame = 0.4):

    # criando/limpando os diretórios pra organizar os arquivos gerados. :P
    os.system('rm imagens_'+politics.nome+'/*')
    os.system('rmdir imagens_'+politics.nome)
    os.mkdir('imagens_'+politics.nome)
    os.system('rm gif_'+politics.nome+'/*')
    os.system('rmdir gif_'+politics.nome)
    os.mkdir('gif_'+politics.nome)
    
    tempo_total_treinamento = politics.tempo_total_treinamento
   
    # os plots funcionam com até 60 bandidos. se você estiver muito empolgado multiplique por mais que 10 rs
    cores = 10*['r','b', 'g', 'c', 'm', 'y']
    
    # para cada tempo decorrido do treinamento estamos colocando na coordenada x qual bandido foi escolhido
    # e se foi uma vitória (cor preenchida) ou uma derrota (dentro em branco).
    for r, j in enumerate(range(tempo_total_treinamento)):

        fig = plt.figure(tight_layout=True, figsize=(15,5))
        gs = gridspec.GridSpec(2, politics.numero_de_bandidos)
        
        ax = fig.add_subplot(gs[0, :])
        plt.xlim(-0.2, tempo_total_treinamento + 0.2)
        plt.ylim(-0.4, politics.numero_de_bandidos - 0.6)

        for i in range(politics.numero_de_bandidos):

            x = [valor for valor in politics.historico[str(i)][0] if valor <= j]
            y = politics.historico[str(i)][1][0:len(x)]
            
            if not epsilon_greedy:
                ax.scatter(x,i*np.ones(len(y)), s=100, edgecolors = cores[i], c = [cores[i] if t==1 else 'w' for t in y]) 
            
            else:
                epsilon_escolhido = [valor for valor in politics.historico[str(i)][0] if valor <= j and politics.historico_epsilon[0][valor]==1]
                y_escolhido = politics.historico[str(i)][1][0:len(epsilon_escolhido)]
                
                epsilon_nao_escolhido = [valor for valor in politics.historico[str(i)][0] if valor <= j and politics.historico_epsilon[0][valor]==0]
                y_nao_escolhido = politics.historico[str(i)][1][0:len(epsilon_nao_escolhido)]
                
                ax.scatter(epsilon_escolhido,i*np.ones(len(y_escolhido)), s=100, edgecolors = cores[i], marker = '.', c = [cores[i] if t==1 else 'w' for t in y_escolhido]) 
                ax.scatter(epsilon_nao_escolhido,i*np.ones(len(y_nao_escolhido)), s=100, edgecolors = cores[i], marker = '*', c = [cores[i] if t==1 else 'w' for t in y_nao_escolhido]) 
            
        for t in range(politics.numero_de_bandidos):
            ax = fig.add_subplot(gs[1, t])
            if len(lista_bandidos)>0:
                ax.plot([lista_bandidos[t],lista_bandidos[t]],[0,8], c='black',alpha = 0.3)
            plt.xlim(-0.02, 1.02)
            plt.ylim(-0.1, 8.1)
            ax.plot([politics.historico_par[j][t],politics.historico_par[j][t]],[0,8], c= cores[t])
            
            if thompson:
                a = 1 + politics.historico_vit[j][t]
                b = 1 + politics.historico_der[j][t]
                
                if j+1<tempo_total_treinamento:
                    if t == np.argmax(np.asarray(politics.historico_estimado_da_rodada[j+1])):
                        ax.scatter([politics.historico_estimado_da_rodada[j+1][t]], [beta.pdf(politics.historico_estimado_da_rodada[j+1][t], a, b)], c =cores[t], s=200)
                    else:
                        ax.scatter([politics.historico_estimado_da_rodada[j+1][t]], [beta.pdf(politics.historico_estimado_da_rodada[j+1][t], a, b)], c =cores[t], s=50)
                    
                x = np.linspace(0,1)
                plt.plot(x, beta.pdf(x, a, b), c= cores[t], lw=3, alpha=0.6)
                
        # salvando a imagem pra cada tempo do treinamento
        plt.savefig('imagens_'+politics.nome+'/historico'+str(j)+'.jpg'.format(r=r))
        plt.close()
    
    gif_path = "gif_"+politics.nome+"/historico.gif"
    frames_path = "imagens_"+politics.nome+"/historico{r}.jpg"

    with imageio.get_writer(gif_path, mode='I', duration=segundos_frame) as writer:
        for r in range(tempo_total_treinamento):
            writer.append_data(imageio.imread(frames_path.format(r=r)))
            
def exemplos_beta():
    alphas = [0, 2, 2, 16, 32, 130]
    betas = [0, 0, 3, 9, 50, 80]
    cores = ['r','b', 'g', 'c', 'm', 'y']
    
    fig = plt.figure(tight_layout=True, figsize=(13,4))
    gs = gridspec.GridSpec(2, 3)
    
    for i in range(6):
        ax = fig.add_subplot(gs[int(i/3),i%3])
        x = np.linspace(0,1)
        plt.xlim(-0.1, 1.1)
        plt.plot(x, beta.pdf(x, 1+alphas[i], 1+betas[i]), lw=3, alpha=1, c = cores[i])
        
        if i!=0:
            ax.plot([alphas[i]/(alphas[i]+betas[i]), alphas[i]/(alphas[i]+betas[i])],[0,beta.pdf(alphas[i]/(alphas[i]+betas[i]), 1+alphas[i], 1+betas[i])], c=cores[i],alpha = 0.7)
        
        plt.title('Sucessos: '+str(alphas[i])+' | Falhas: '+str(betas[i]))
        

