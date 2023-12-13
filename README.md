# Diabetes Health Indicators - Data Science & Machine Learning
### [Autor: Lucas Barbosa Nascimento](https://github.com/Lucasbnas435)

Nesse projeto, encontra-se uma análise estatística dos dados do Centers for Disease Control and Prevention (CDC) e, a partir disso, é gerado um modelo de Machine Learning que realiza classificação binária (binary classification), prevendo se determinada pessoa **não é diabética (0)** ou encontra-se numa condição de **pré-diabética/diabética (1)**.

## Dataset

O Behavioral Risk Factor Surveillance System (BRFSS) é uma pesquisa telefônica relacionada à saúde realizada anualmente pelo CDC. A cada ano, são coletadas respostas de mais de 400.000 norte-americanos sobre comportamentos pessoais, fatores de risco, doenças crônicas e uso de serviços preventivos. Neste projeto, utilizou-se um conjunto de dados resultante da pesquisa conduzida em 2015, contendo 253.680 respostas com 21 características indicadas diretamente pelos participantes ou calculadas com base em suas respostas individuais.

Variáveis contidas no dataset:

- Diabetes_binary: 0 = não diabético; 1 = pré-diabético ou diabético
- HighBP: 0 = sem diagnóstico de hipertensão; 1 = possui hipertensão
- HighChol: 0 = sem diagnóstico de colesterol alto; 1 = apresenta colesterol alto
- CholCheck: 0 = não fez exame de colesterol nos últimos 5 anos; 1 = fez esse exame nos últimos 5 anos
- BMI: Índice de Massa Corporal
- Smoker: 0 = não fumou, pelo menos, 100 cigarros ao longo da vida; 1 = já fumou 100 cigarros ou mais
- Stroke: 0 = nunca teve diagnóstico de AVC; 1 = já teve diagnóstico de AVC
- HeartDiseaseorAttack: 0 = nunca diagnosticado com doença cardíaca coronária ou infarto do miocárdio: 1 = possui tal doença ou já sofreu infarto do miocárdio
- PhysActivity: 0 = não realizou atividade física nos últimos 30 dias (excluindo-se atividades relacionadas a trabalho); 1 = realizou atividade física nesse período
- Fruits: 0 = não consome fruta pelo menos uma vez por dia; 1 = consome fruta pelo menos uma vez por dia
- Veggies: 0 = não consome vegetais pelo menos uma vez por dia; 1 = consome vegetais pelo menos uma vez por dia
- HvyAlcoholConsump: 0 = não consome grande quantidade de álcool; 1 = consome (pelo menos 14 drinks por semana para homens adultos e 7 drinks por semana para mulheres adultas)
- AnyHealthcare: 0 = não possui cobertura de saúde; 1 = possui (inclui seguros de saúde, planos pré-pagos e outros tipos)
- NoDocbcCost: 0 = não deixou de visitar um médico por motivos financeiros no último ano; 1 = deixou de fazê-lo por motivos financeiros nesse período
- GenHlth: "Numa escala de 1 (excelente) a 5 (ruim), como está a sua saúde em geral?"
- MentHlth: quantidade de dias com má saúde mental no último mês
- PhysHlth: quantidade de dias com má saúde física (incluindo doenças e lesões) no último mês
- DiffWalk: 0 = não possui grande dificuldade para andar ou subir escadas; 1 = possui uma dessas dificuldades
- Sex: 0 = feminino; 1 = masculino
- Age: 13 categorias de idade (18-24 anos, 25-29, ..., 75-79, 80 anos ou mais)
- Education: 6 níveis de escolaridade completa (1 = nunca foi à escola; 6 = completou o College)
- Income: 8 níveis de renda anual daquele domicílio (1 = menos de US$ 10000; 8 = US$ 75000 ou mais)

## Desenvolvimento do Projeto

A aplicação foi desenvolvida em Python 3.10, utilizando um [Jupyter Notebook](https://jupyter.org/) hospedado e executado na plataforma [Google Colaboratory](https://colab.research.google.com/). Ademais, foram utilizadas as bibliotecas [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [SciPy](https://docs.scipy.org/doc/scipy/), [Imbalanced-learn](https://imbalanced-learn.org/stable/), [LightGBM](https://lightgbm.readthedocs.io/en/stable/), [Scikit-learn](https://scikit-learn.org/stable/), [XGBoost](https://xgboost.readthedocs.io/en/stable/) e [Joblib](https://joblib.readthedocs.io/en/stable/).

O notebook diabetes_health_indicators.ipynb apresenta, inicialmente, a exploração e a visualização dos dados, expondo informações e gráficos diversos que atuam como referências fundamentais para a Análise Exploratória. Ao longo do código, são feitos comentários exibindo ideias obtidas a partir do conjunto de dados e a linha de raciocínio percorrida na tomada de decisões.

Em seguida, tendo como base as observações e conclusões alcançadas na etapa anterior, realiza-se o Feature Scaling (normalização usando o Min-Max Scaler, visto que as variáveis envolvidas nesse processo não estavam normalmente distribuídas). Então, prosseguindo com a abordagem estatística do dataset, é feito o Teste do Qui-quadrado de Pearson entre as variáveis categóricas e Diabetes_binary (target variable), evidenciando que a hipótese nula (afirmação de que os valores comparados são estatisticamente independentes) foi recusada em todos os casos. Assim, não havendo eliminações por conta desse teste de hipótese, as colunas menos correlacionadas com o alvo foram excluídas.

Na sequência, sabendo que os dados estavam desbalanceados (84,71% de não diabéticos; 15,29% de pré-diabéticos ou diabéticos), efetuou-se oversampling e, logo após, a divisão do dataset entre parte de treinamento e parte de testes. Completando isso, executa-se o treinamento dos modelos com os algoritmos XGBoost, Decision Tree, Random Forest e LightGBM. Por fim, tais modelos são salvos e exportados, viabilizando seu compartilhamento e uso em outros locais.

## Resultados

Com o algoritmo XGBoost, alcançou-se uma acurácia de **78,56%**. Esse dado, a matriz de confusão e o classification report estão expostos logo abaixo:

![](https://github.com/Lucasbnas435/Diabetes-Health-Indicators/blob/master/src/docs/XGBoost_results.png?raw=true)

Empregando-se Decision Tree, obteve-se uma acurácia de **91,50%**. Esse dado, a matriz de confusão e o classification report estão expostos logo abaixo:

![](https://github.com/Lucasbnas435/Diabetes-Health-Indicators/blob/master/src/docs/DecisionTree_results.png?raw=true)

Com o algoritmo Random Forest, por sua vez, atingiu-se uma acurácia de **93,96%**. Esse dado, a matriz de confusão e o classification report estão expostos logo abaixo:

![](https://github.com/Lucasbnas435/Diabetes-Health-Indicators/blob/master/src/docs/RandomForest_results.png?raw=true)

O modelo produzido com LightGBM apresenta acurácia de **74,59%**. Esse dado, a matriz de confusão e o classification report estão expostos logo abaixo:

![](https://github.com/Lucasbnas435/Diabetes-Health-Indicators/blob/master/src/docs/LightGBM_results.png?raw=true)

## Estrutura de Pastas

```
.
├── README.md
└── src
    ├── dataset
    │   └── diabetes_binary_health_indicators_BRFSS2015.csv
    ├── docs
    │   ├── DecisionTree_results.png
    │   ├── LightGBM_results.png
    │   ├── RandomForest_results.png
    │   └── XGBoost_results.png
    ├── models
    │   ├── DecisionTreeModel.joblib
    │   ├── LightGBMModel.pkl
    │   └── XGBoostModel.json
    └── notebook
        └── diabetes_health_indicators.ipynb
```

## Fonte dos Dados

Os dados utilizados estão disponíveis em:

- https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

## Contato

Muito obrigado por acessar esse projeto!

Vamos nos conectar?

- GitHub: https://github.com/Lucasbnas435
- LinkedIn: https://linkedin.com/in/lucasbnas
- E-mail: lucasbnas435@gmail.com
