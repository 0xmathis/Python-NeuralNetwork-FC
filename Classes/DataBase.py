from typing import Sequence

from sympy import N

from Matrice import Matrice


class DataBase:
    def __init__(self, shape: tuple[int, ...], biasesPath: str, weightsPath: str):
        self.__shape: Sequence = shape
        self.__biasesPath = biasesPath
        self.__weightsPath = weightsPath

        open(biasesPath, 'w')
        open(weightsPath, 'w')

        if self.isFileEmpty(biasesPath) and self.isFileEmpty(weightsPath):
            self.biasesList, self.weightsList = [], []
            self.biasesFileFull, self.weightsFileFull = False, False
        else:
            self.biasesList, self.weightsList = self.__getBiasesListFromFile(), self.__getWeightsListFromFile()
            self.biasesFileFull, self.weightsFileFull = True, True

    def __biasesMatricesToFile(self, biasesList: list[Matrice]) -> None:
        with open(self.__biasesPath, 'w') as f:
            for matrice in biasesList:
                for i in range(matrice.getRows()):
                    f.write(str(matrice[(i, 0)]))
                    if matrice != biasesList[-1] or i != matrice.getRows() - 1:
                        f.write('\n')

    def __weightsMatricesToFile(self, weightsList: list[Matrice]):
        with open(self.__weightsPath, 'w') as f:
            for matrice in weightsList:
                for i in range(matrice.getRows()):
                    for j in range(matrice.getColumns()):
                        f.write(str(matrice[(i, j)]))
                        if matrice != weightsList[-1] or i != (matrice.getRows() - 1) or j != (matrice.getColumns() - 1):
                            f.write('\n')

    def getBiases(self, toLayer: int) -> Matrice:
        return self.biasesList[toLayer]

    def __getBiasesListFromFile(self) -> list[Matrice]:
        biases = []

        with open(self.__biasesPath, 'r') as f:
            for n in range(1, len(self.__shape)):
                matrice = [[float(f.readline())] for _ in range(self.__shape[n])]

                biases.append(Matrice(matrice).map(N))

        return biases

    def getWeights(self, toLayer: int) -> Matrice:
        return self.weightsList[toLayer]

    def __getWeightsListFromFile(self) -> list[Matrice]:
        weights = []

        with open(self.__weightsPath, 'r') as f:
            for n in range(1, len(self.__shape)):
                matrice = [[float(f.readline()) for _ in range(self.__shape[n - 1])] for _ in range(self.__shape[n])]

                weights.append(Matrice(matrice).map(N))

        return weights

    def isBiasesFileFull(self) -> bool:
        return self.biasesFileFull

    def isWeightsFileFull(self) -> bool:
        return self.weightsFileFull

    @staticmethod
    def isFileEmpty(filePath: str) -> bool:
        with open(filePath, 'r') as f:
            return len(f.readlines()) == 0

    def toFile(self, biasesList, weightsList) -> None:
        self.__biasesMatricesToFile(biasesList)
        self.__weightsMatricesToFile(weightsList)
