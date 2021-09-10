
---

## Workshop
**Pytorch lightning and tensorboard for better experiment management**

---

## Why Lightning ?

**Engineering** vs **Research**
- Infra
- managing logs
- organizing code
- saving/loading models

(popular, recommended on conf like neurips for reproducibility)

---

## Why tensorboard ?
A lot of alternatives: 
- MLflow
- Weight and biases
- Neptune
- sacred ...

free, default, powerful, initial focus on training metrics

---

## Plan 
* Lightning code walkthrough
* First training and testing
* Adding a test diagnostic to tensorboard
	- logging hyperparams and metrics
	- logging a confusion matrix
	- logging best and worst prediction for one class
* Adding more technical logging to tensorboard
	- logging the model graph
	- logging the learning rate 
	- Logging the profile of a training step 
