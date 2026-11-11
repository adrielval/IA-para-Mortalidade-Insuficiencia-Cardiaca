Modelo de aprendizado de máquina supervisionado para prever o risco de óbito em pacientes com insuficiência cardíaca a partir de dados clínicos.

Usei o dataset público Heart Failure Clinical Records (UC Irvine Machine Learning Repository), com registros de 299 pacientes e variáveis como idade, fração de ejeção, creatinina sérica, sódio sérico, plaquetas, pressão alta, diabetes, anemia e tabagismo.

O que o projeto faz:
🔹 Carrega e inspeciona os dados com Pandas
🔹 Faz o pré-processamento com codificação one-hot das variáveis
🔹 Separa os dados em treino (70%) e teste (30%)
🔹 Treina uma Árvore de Decisão (critério de entropia, profundidade máxima 3) com Scikit-learn
🔹 Gera a visualização da árvore com Matplotlib, mostrando quais fatores clínicos mais pesam na previsão

Escolhi a Árvore de Decisão pela interpretabilidade: em saúde, entender por que o modelo chegou a uma conclusão importa tanto quanto a previsão em si. A árvore mostra o caminho de decisão de forma visual, o que facilita a leitura até por quem não é da área de dados.

Projeto desenvolvido como avaliação na graduação em Ciência da Computação.
