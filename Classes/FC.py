from random import randint
from time import time
from typing import Any, Sequence

from matplotlib.pyplot import plot, show, subplot, ylabel
from sympy import N

from Classes.DataBase import DataBase
from Classes.Layer import Layer
from Matrice import Matrice
from time import localtime


class FC:
    def __init__(self, shape: tuple[int, ...], learningRate: float, momentumRate: float, biasesPath: str = 'Database\\biases.txt', weightsPath: str = 'Database\\weights.txt'):
        self.__learningRate: float = learningRate
        self.__momentumRate: float = momentumRate

        self.__database: DataBase = DataBase(shape, biasesPath=biasesPath, weightsPath=weightsPath)

        self.__dataSet: list[Matrice] = []
        self.__targetsSet: list[Matrice] = []

        self.__layers: list[Layer] = []
        for i in range(1, len(shape)):
            if self.__database.isBiasesFileFull() and self.__database.isWeightsFileFull():
                self.__layers += [Layer(shape, i, self.__learningRate, self.__momentumRate, biases=self.__database.getBiases(i - 1), weights=self.__database.getWeights(i - 1))]
            else:
                self.__layers += [Layer(shape, i, self.__learningRate, self.__momentumRate)]

        if not (self.__database.isBiasesFileFull() and self.__database.isWeightsFileFull()):
            self.__toFile()

    def __backPropagation(self, inputs: Matrice, targets: Matrice) -> None:
        # calcul des deltas
        for i in range(len(self.__layers) - 1, -1, -1):
            if i == len(self.__layers) - 1:
                self.__layers[i].setOutputDeltas(targets)
            else:
                self.__layers[i].setDeltas(self.__layers[i + 1].getWeights(), self.__layers[i + 1].getDeltas())

        # calcul des dCosts
        for i in range(len(self.__layers) - 1, -1, -1):
            self.__layers[i].setdCostsdWeights(inputs if i == 0 else self.__layers[i - 1].getOutput())

        # tuning
        for i in range(len(self.__layers)):
            self.__layers[i].tuning()

    @staticmethod
    def __date_in(sec: float) -> str:
        dico_days = {
            0: 'lun', 1: 'mar',
            2: 'mer', 3: 'jeu',
            4: 'ven', 5: 'sam',
            6: 'dim'
            }
        dico_months = {
            1: 'janvier', 2: 'fevrier',
            3: 'mars', 4: 'avril',
            5: 'mai', 6: 'juin',
            7: 'juillet', 8: 'aout',
            9: 'septembre', 10: 'octobre',
            11: 'novembre', 12: 'decembre',
            }

        final = localtime(time() + sec)

        return f'{dico_days[final.tm_wday]} {final.tm_mday} {dico_months[final.tm_mon]} {final.tm_hour}:{final.tm_min}:{final.tm_sec}'

    @staticmethod
    def __from_secondes(sec: float) -> str:
        final = ''
        for i, txt in ((2, 'h'), (1, 'm'), (0, 's')):
            val = int((sec % (60 ** (i + 1))) // (60 ** i))
            if val != 0:
                final += str(val) + txt
        milli = round((sec % 1) * 1000)
        if milli != 0:
            final += f'{milli}ms'

        return final or '0s'

    def __evaluate_time(self, iterations: int) -> str:
        temp2 = time()
        choix = randint(0, len(self.__dataSet) - 1)
        data = self.__dataSet[choix]
        target = self.__targetsSet[choix]
        self.__feedForward(data)
        self.__backPropagation(data, target)
        time_ff_dp = time() - temp2

        time_total = time_ff_dp * iterations

        return f'Temps total évalué : {self.__from_secondes(time_total)}\nFin évalué : {self.__date_in(time_total)}'

    def __feedForward(self, data: Matrice) -> Matrice:
        for layer in self.__layers:
            data = layer.feedForward(data)

        return data

    def __toFile(self) -> None:
        biasesList = [layer.getBiases() for layer in self.__layers]
        weightsList = [layer.getWeights() for layer in self.__layers]

        self.__database.toFile(biasesList=biasesList, weightsList=weightsList)

    def guess(self, data: list) -> tuple[Any]:
        data_ = Matrice([data]).T
        guess = self.__feedForward(data_)
        guess_ = guess.T.toList()[0]

        return tuple(N(x, 10) for x in guess_)

    def getLayers(self) -> list[Layer]:
        return self.__layers

    def setDataSet(self, dataSet: Sequence[Sequence]) -> None:
        self.__dataSet = [Matrice([data]).T for data in dataSet]

    def setTargetsSet(self, targetsSet: Sequence[Sequence]) -> None:
        self.__targetsSet = [Matrice([target]).T for target in targetsSet]

    def trainFromDataInObject(self, nbIterations: int, freq: int, graph: bool) -> None:
        # iteration est le nombre d'entrainements à faire
        # freq est la frequence à laquelle sauvegarder la base de données

        yErrorPlot = []
        yTimePlot = []
        start = time()

        for i in range(1, nbIterations + 1):
            temp = time()

            choix = randint(0, len(self.__dataSet) - 1)
            data: Matrice = self.__dataSet[choix]
            target: Matrice = self.__targetsSet[choix]

            self.__feedForward(data)
            self.__backPropagation(data, target)

            if i % freq == 0:
                self.__toFile()

            temp2 = time() - temp

            print(f'{i} :', self.__from_secondes(temp2))

            if graph:
                yErrorPlot.append(abs(self.getLayers()[-1].getOutputCosts()))
                yTimePlot.append(temp2)

        total = time() - start
        print('Temps total :', self.__from_secondes(total), '\n')

        if graph:
            self.showGraph(nbIterations, yErrorPlot, yTimePlot)

    def trainFromDataInObjectWhileCostAboveMax(self, maxCost: float, graph: bool) -> None:
        # iteration est le nombre d'entrainements à faire
        # freq est la frequence à laquelle sauvegarder la base de données

        yErrorPlot = []
        yTimePlot = []
        start = time()
        nbIterations = 1
        cost = 1

        while cost > maxCost:
            temp = time()

            choix = randint(0, len(self.__dataSet) - 1)
            data: Matrice = self.__dataSet[choix]
            target: Matrice = self.__targetsSet[choix]

            self.__feedForward(data)
            self.__backPropagation(data, target)

            cost = abs(self.getLayers()[-1].getOutputCosts())

            temp2 = time() - temp

            print(f'{nbIterations} : {self.__from_secondes(temp2)} ; Cost : {N(cost, 8)}')

            if graph:
                yErrorPlot.append(cost)
                yTimePlot.append(temp2)

            nbIterations += 1

        total = time() - start
        print('Temps total :', self.__from_secondes(total), '\n')

        self.__toFile()

        if graph:
            self.showGraph(nbIterations, yErrorPlot, yTimePlot)

    def trainFromExternalData(self, dataMatrice: Matrice, targetMatrice: Matrice, iteration: int, freq: int):
        start = time()

        self.__feedForward(dataMatrice)
        self.__backPropagation(dataMatrice, targetMatrice)

        if iteration % freq == 0:
            self.__toFile()

        print(f'{iteration} :', self.__from_secondes(time() - start))

    @staticmethod
    def showGraph(nbIterations, yErrorPlot, yTimePlot):
        subplot(2, 1, 1)
        ylabel('Norme du cout')
        plot(list(range(1, nbIterations)), yErrorPlot, linewidth=0.2)
        subplot(2, 1, 2)
        ylabel('Durée de chaque itération (ms)')
        plot(list(range(1, nbIterations)), yTimePlot)
        show()
