# Trabalho Final -Aprendizado de Máquina

> Fundação Getúlio Vargas RJ - Escola de Matemática Aplicada<br>
> Graduação em Ciência de Dados e Inteligência Artificial<br>
> Disciplina: Aprendizado de Máquina<br>
> Aluno: Gianlucca Devigili<br>

# Introdução:

# 1. Os dados

Para a execução do presente trabalho, utilizei o clássico dataset do [titanic](https://www.kaggle.com/competitions/titanic/data).  Uma pequena amostra dos dados pode ser vista na imagem abaixo:

![[Pasted image 20220612233403.png]]

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
+ **Embarked:** `str` `{C = Cherbourg, Q = Queenstown, S = Southampton}` - indica o porto de embarque do passageiro. O dataset, obviamente, apenas considera passageiros que realmente embarcaram.



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