# Trabalho Final -Aprendizado de Máquina

> Fundação Getúlio Vargas RJ - Escola de Matemática Aplicada<br>
> Graduação em Ciência de Dados e Inteligência Artificial<br>
> Disciplina: Aprendizado de Máquina<br>
> Aluno: Gianlucca Devigili<br>

# 1. Os dados
Para a execução do presente trabalho, utilizei o clássico dataset do [titanic](https://www.kaggle.com/competitions/titanic/data).  Uma pequena amostra dos dados pode ser vista na imagem abaixo:

![[Pasted image 20220612233403.png]]

## 1.1. Dicionário de Dados
O dataset conta com:
+ 5 colunas categóricas (tipo: `object`)
+ 4 colunas discretas (tipo: `int 64`)
+ 2 colunas contínuas (tipo: `float 64`)

As colunas do dataset são as seguintes:
+ **Survived:** `bool` - `{0, 1}` - a coluna *target* do dataset. Indica se o passageiro sobreviveu ($1$) ou não ($0$);
+ **PClass:** `int` - `{1, 2, 3}` - indica a classe do passageiro, sendo primeira classe a mais alta e a terceira a mais baixa;
+ **Name:** `str` - nome do passageiro. A coluna não foi utilizada para a realização do trabalho por conta da grande complexidade de tratar tal coluna de forma efetiva. Uma maneira da qual estes dados poderiam ser utilizados é a identificação por sobrenome e uma classificação em famílias, contudo pessoas com sobrenomes iguais porém de famílias diferentes poderiam ser impossíveis de distinguir dado as informações presentes no dataset, além da geração de uma grande quantidade de *features* novas que não garantiriam melhora na performance do modelo, além de acarretar em aumento da complexidade do mesmo.
+ **Sex:** `str` - `{"male", "female"}` - indica o sexo do passageiro (feminino ou masculino);
+ **SibSp:** `ìnt` - `0 à 8` - indica a quantidade de irmãos e cônjuges a bordo do navio;
+ **Parch:** `int` - `0 à 9` - indica o número de pais e filhos a bordo do navio;
+ **Ticket:** `tr` - indica o número do bilhete de embarque. A coluna não foi utilizada  por conta da grande diversidade de formas de identificação do número dos bilhetes, o que poderia gerar dificuldades de tratar corretamente os dados;
+ **Fare:** `float` - indica o preço pago na passagem;
+ **Cabin:** `str` - indica a cabine do passageiro. Por conta da grande quantidade de dados faltantes, a coluna não foi utilizada no dataset.
+ **Embarked:** `str` - `{C = Cherbourg, Q = Queenstown, S = Southampton}` - indica o porto de embarque do passageiro. O dataset, obviamente, apenas considera passageiros que realmente embarcaram.

## 1.2. Tratamento dos Dados
O dataset original, como tem o propósito de servir como dataset para uma competição do *website Kaggle*, está subdividido em 3 datasets: `train.csv`, `test.csv` e `gender_submission_csv`, sendo este último contendo a coluna `Survived` referente aos dados do dataset de teste. Preferi juntar todos os dados em um único dataset e então dividir novamente em treino e teste por dois motivos principais: precisei realizar algumas transformações e tratamentos nos dados de algumas colunas e a concatenação dos datasets em um só tornou tal processo mais efetivo e a possibilidade de uma análise exploratória completa com os dados dos passageiros do navio.

Primeiramente eliminei as colunas `Name` e `Ticket` pelos motivos já discutidos acima, bem como a coluna `Cabin` pela grande quantidade de dados faltantes na mesma.

Agora, analisando as informações do dataset mostradas pelo método  `pandas.DataFrame.info()`, temos:
![[Pasted image 20220613002557.png]]
evidenciando a necessidade de tratamento de dados faltantes nas colunas `Age`, `Fare` e `Embarked`.

Para a coluna `Age`, substituí os dados faltantes pela idade média dos passageiros: 30 anos.

Para a coluna `Fare`, substituí os dados faltantes pela moda da coluna, no caso o porto de Southampton (`"S"`), que onde provavelmente foi o embarque de fato de tais passageiros já que existem $914$ embarques neste porto, contra $270$ em Cherbourg e $123$ em Queenstown.

Para a coluna `Fare`, de modo a deixar ela mais próxima do que poderia ser o dado real que não está presente na base, utilizei o fato de que a classe do passageiro tem grande influência no valor de sua passagem, o que pode ser evidenciado na seguinte visualização:
![[Pasted image 20220613003554.png]]
portanto substituí os dados faltantes pelo preço médio da passagem referente à classe do passageiro que não tinha este dado.

Um problema encontrado na coluna `Fare` que pode ser visto no *boxplot* acima é a presença de valores estremos no dataset. Ao todo são 4 valores com tal problema e todos eles apresentam o valor de $\$512.3292$, indicando que provavelmente se trata de um *outlier*. Da mesma forma que fiz com o passageiro que não possuía um valor de passagem, substituí os dados extremos pela média do valor da passagem agrupada pela classe dos passageiros, que neste caso em particular era a primeira classe.



## Detalhes Técnicos:
O trabalho foi realizado utilizando *jupyter notebook*, versão `7.1.0` com um kernel `python` em sua versão `3.8.10 64-bit`.

Abaixo seguem as bibliotecas utilizadas:

**Modelos**
```
sklearn 1.0.1
```

**Manipulação de Dados**
```
pandas 1.3.5
numpy 1.21.4
scipy 1.7.3
```

**Visualização**
```
seaborn 0.11.2
matplotlib 3.5.1
```