# Aspect-Based Sentiment Analysis
Aspect-based sentiment analysis is the task of identifying the aspect terms of a given sentence and the related opinion polarity.

## Aspect Term Identification

In order to solve this problem I tried two approaches:
- BiLSTM based models
- Bert based models

For all the following models I tried to both use a binary classification and a multi class classification in order to classify aspect terms. In the binary classification each term classified as `1` is considered an aspect term and, in order to deal with aspect terms formed by multiple tokens, I decided to aggregate consecutive terms classified as `1` into a single aspect term.
Instead with the multi class classification I used three labels: `B`, `I` and `O`. `B` is the class for the token that are at the beginning of an aspect term, `I` is the class for the tokens that are inside an aspect term but that are not the first ones and `O` is the class for the tokens that are not inside an aspect term. With the latter approach I am able to handle sentences where there are consecutive aspect terms.

### BiLSTM Model
With this approach I used the Glove 840B 300d (Pennington et al., 2014) pretrained embeddings and I have managed the OOV terms with a special token `UNK` associated with a random vector as embedding. This model as said before is based on a LSTM and in particular is based on a BiLSTM in order to obtain contextualized embeddings that depend on both the precedent and subsequent terms. The BiLSTM takes as input the embeddings of the terms of the considered sentence and outputs the contextualized embeddings of each term. Each contextualized embedding than is passed through a multi layer perceptron (MLP) in order to obtain a classification for each term.

### BERT Models
The following approaches are all similar and change in the way of handling terms associated to more than one word piece. All this models use the BERT tokenizer in order to obtain the word pieces of the sentence and their encoding. The encoded sentence than is passed through the pretrained BERT-large-cased model in order to ob- tain the contextualized embedding for each word piece. The contextualized embeddings of the word pieces are used in order to obtain the contextualized embeddings of the terms though the combination techniques that I will explain below. Each term contextualized embedding than passes through a MLP in order to classify it.
The techniques used in order to combine the contextualized embeddings of the word pieces into the contextualized embedding of the associated term are the following ones:
- SUM: take the sum of the contextualized em- beddings of the word pieces that form the con- sidered term.
- AVG: take the average of the contextualized embeddings of the word pieces that form the considered term.
- MAX: take the max element wise of the con- textualized embeddings of the word pieces that form the considered term.
- FIRST: consider only the first word piece of the considered term.

## Aspect Term Polarity Classification
In this section I am going to approach the problem of the aspect term polarity classification. In order to tackle this problem I have developed two approaches based on BERT. The first one makes use of the combination techniques explained in the precedent task, while the second one uses a special token in order to represent the aspect term.

### Special Token Approach 
With this approach we have that in each sentence we replace the aspect term with a new token `<target-term>` that is initially associated to a random embedding that the BERT model will learn during the train. Each sentence is associated through the BERT tokenizer to the ids of the word pieces in the sentence. Than with the BERT-large-uncased model we obtain the contextualized embeddings of each word piece. The contextualized embedding of the `<target-term>` token is passed through a MLP and classified over the four sentiment classes.

### Without Special Token Approach
In this case, rather than replacing the aspect term with a special token I decided to let the model take as input the entire original sentence and than combine the word pieces of the aspect term with two of the precedent explained combination methods: `MAX` and `AVG` that where the most efficient ones in the precedent task. After this also in this case, the contextualized embedding of the aspect term is passed through an MLP and classified over the four sentiment classes.

## Results
The tested model has been obtained by the concatenation of the model with the `MAX` technique for combining the word pieces for obtaining the aspect terms and the model with the special token in order to classify the polarity of the predicted aspect terms. This model obtained a Macro F1 of `0.538` and a micro F1 of `0.639`.

## You can find more information in the `report.pdf` where are shown and compared the results obtained for the two tasks of the different models proposed.

#### Instructor
* **Roberto Navigli**
	* Webpage: http://wwwusers.di.uniroma1.it/~navigli/

#### Teaching Assistants
* **Cesare Campagnano**
* **Pere-Lluís Huguet Cabot**

#### Course Info
* http://naviglinlp.blogspot.com/

## Requirements

* Ubuntu distribution
	* Either 19.10 or the current LTS are perfectly fine
	* If you do not have it installed, please use a virtual machine (or install it as your secondary OS). Plenty of tutorials online for this part
* [conda](https://docs.conda.io/projects/conda/en/latest/index.html), a package and environment management system particularly used for Python in the ML community

## Notes
Unless otherwise stated, all commands here are expected to be run from the root directory of this project

## Setup Environment

As mentioned in the slides, differently from previous years, this year we will be using Docker to remove any issue pertaining your code runnability. If test.sh runs
on your machine (and you do not edit any uneditable file), it will run on ours as well; we cannot stress enough this point.

Please note that, if it turns out it does not run on our side, and yet you claim it run on yours, the **only explanation** would be that you edited restricted files, 
messing up with the environment reproducibility: regardless of whether or not your code actually runs on your machine, if it does not run on ours, 
you will be failed automatically. **Only edit the allowed files**.

To run *test.sh*, we need to perform two additional steps:
* Install Docker
* Setup a client

For those interested, *test.sh* essentially setups a server exposing your model through a REST Api and then queries this server, evaluating your model.

### Install Docker

```
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

Unfortunately, for the latter command to have effect, you need to **logout** and re-login. **Do it** before proceeding. For those who might be
unsure what *logout* means, simply reboot your Ubuntu OS.

### Setup Client

Your model will be exposed through a REST server. In order to call it, we need a client. The client has already been written
(the evaluation script) but it needs some dependecies to run. We will be using conda to create the environment for this client.

```
conda create -n nlp2021-hw2 python=3.7
conda activate nlp2021-hw2
pip install -r requirements.txt
```

## Run

*test.sh* is a simple bash script. To run it:

```
conda activate nlp2021-hw2
bash test.sh data/restaurants_dev.json
```

Actually, you can replace *data/dev.jsonl* to point to a different file, as far as the target file has the same format.

If you hadn't changed *hw2/stud/model.py* yet when you run test.sh, the scores you just saw describe how a random baseline
behaves. To have *test.sh* evaluate your model, follow the instructions in the slide.
