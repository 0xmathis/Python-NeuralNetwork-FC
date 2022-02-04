from sympy import exp, N

from Matrice import Matrice


class Layer:
    def __init__(self, shape: tuple[int, ...], column: int, learningRate: float, momentumRate: float, biases: Matrice = None, weights: Matrice = None):
        self.__shape: tuple[int, ...] = shape
        self.__column: int = column
        self.__learningRate: float = learningRate
        self.__momentumRate: float = momentumRate

        if biases is None and weights is None:
            self.__biases = Matrice.random(self.__shape[self.__column], 1, -1, 1, float).map(N)
            self.__weights = Matrice.random(self.__shape[self.__column], self.__shape[self.__column - 1], -1, 1, float).map(N)
        else:
            self.__biases = biases
            self.__weights = weights

        self.__input: Matrice = Matrice.vide(self.__shape[self.__column - 1], 1)
        self.__weightedSum: Matrice = Matrice.vide(self.__shape[self.__column - 1], 1)
        self.__output: Matrice = Matrice.vide(self.__shape[self.__column], 1)
        self.__deltas: Matrice = Matrice.vide(self.__shape[self.__column], 1)
        self.__dCost_dWeights: Matrice = Matrice.vide(self.__shape[self.__column], self.__shape[self.__column - 1])
        self.__previousDeltaBiases: Matrice = Matrice.vide(self.__shape[self.__column], 1)
        self.__previousDeltaWeights: Matrice = Matrice.vide(self.__shape[self.__column], self.__shape[self.__column - 1])

    def __d_sigmoid_dx(self, y: float) -> float:
        return N(self.__sigmoid(y) * (1 - self.__sigmoid(y)))

    @staticmethod
    def __sigmoid(x: float) -> float:
        return N(1 / (1 + exp(-x)))

    def feedForward(self, data: Matrice) -> Matrice:
        self.__input = data
        self.__weightedSum = self.__weights * self.__input + self.__biases
        self.__output = self.__weightedSum.map(self.__sigmoid)

        return self.__output

    def getBiases(self) -> Matrice:
        return self.__biases

    def getDeltas(self) -> Matrice:
        return self.__deltas

    def getOutput(self) -> Matrice:
        return self.__output

    def getOutputCosts(self, targets: Matrice) -> Matrice:
        errors = self.__output - targets

        return errors.hp(errors) * 0.5

    def getWeightedSum(self) -> Matrice:
        return self.__weightedSum

    def getWeights(self) -> Matrice:
        return self.__weights

    def setdCostsdWeights(self, previousLayerOutputs: Matrice) -> None:  # pas dCostsdBiases car dCostsdBiases = deltas
        self.__dCost_dWeights = self.__deltas * previousLayerOutputs.T

    def setDeltas(self, nextLayerWeights: Matrice, nextLayerDeltas: Matrice) -> None:
        self.__deltas = (nextLayerWeights.T * nextLayerDeltas).hp(self.__weightedSum.map(self.__d_sigmoid_dx))

    def setOutputDeltas(self, targets: Matrice) -> None:
        errors = self.__output - targets
        self.__deltas = .5 * errors.hp(self.__weightedSum.map(self.__d_sigmoid_dx))

    def tuning(self) -> None:
        self.__biases -= self.__learningRate * self.__momentumRate * self.__deltas - (1 - self.__momentumRate) * self.__previousDeltaBiases
        self.__weights -= self.__learningRate * self.__momentumRate * self.__dCost_dWeights - (1 - self.__momentumRate) * self.__previousDeltaWeights

        self.__previousDeltaBiases = self.__deltas.copy()
        self.__previousDeltaWeights = self.__dCost_dWeights.copy()
