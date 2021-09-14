---
title: Experiment management with lightning and tensorboard
tags: Talk, pytorch lightning, tensorboard
description: Oceanix workshop.
slideOptions:
  spotlight:
    enabled: true
---

## Workshop

**Pytorch lightning and tensorboard for better experiment management**

---


## Why

Demonstrate how the right tools can make your life easier 
https://media.giphy.com/media/0qaV5zc6KBMxrnELxK/giphy.gif

---

## Why Lightning ?

**Engineering** vs **Research** : Spending time on what matters
- Infra (cpu, gpu, cluster)
- managing logs
- organizing code (software engineering)
- saving/loading models

---

## Why tensorboard ?
Tracking and comparing results of different experiments 

**A demo is worth a thousand words**

---

## Plan 

* Lightning code walkthrough
* Workshop exercises description
* To your keyboards


---

### Lightning code walkthrough

```python
class MNISTModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)



# Init our model
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# Initialize a trainer
trainer = Trainer(
    max_epochs=3,
)

# Train the model 
trainer.fit(mnist_model, train_loader)
```

---

## Let's start

The repo for the workshop:

https://github.com/quentinf00/lightning-tensorboard-workshop

