## **Image classification**

**1.dataset**

Cifar10

**2.Operating environment**

GPU

Python 3.6.9

torch 1.9.1+cu111

torchvision
0.10.1+cu111

**3.**Neural network model****

VGG19

**4.experiment steps**

Run train.py to reduce model loss and then run test.py to test the accuracy of model training.

**5.Tunning hyperparameters**

The size of the epoch determines the training of several generations of dataset.Bach-size decides how many images can be trained once.

*Training result or test result:*

| epoch | batch_size | batch_norm | accuracy |
| - | - | - | - |
| 1 | 128 | unset | 0.3310 |
| 1 | 128 | set | 0.5416 |
| 3 | 128 | set | 0.6315 |
| 10 | 128 | set | 0.8015 |
| 20 | 128 | set | 0.8314 |
| 30 | 128 | set | 0.8483 |

| epoch | batch_size | optimizer | lr | lastloss |
| - | - | - | - | - |
| 1 | 64 | Adam | 1e-3 | 1.78 |
| 1 | 64 | Adam | 1e-4 | 1.11 |
| 1 | 64 | Adam | 1e-5 | 1.78 |
| 1 | 128 | Adam | 1e-3 | 1.76 |
| 1 | 128 | Adam | 1e-4 | 1.11 |
| 1 | 128 | Adam | 1e-5 | 1.60 |
| 1 | 128 | SGD | 1e-3 | 1.74 |
| 1 | 128 | SGD | 1e-4 | 1.73 |
| 1 | 128 | SGD | 1e-5 | 2.32 |

**6.Discovery**

From the table1,

⑴we can find that the more iterations under
certain conditions,the higher the accuracy of model classification. It is foreseeable
that when the epoch reaches a suitable value, the accuracy of the test will
tend to 1.From the other side of the table,the accuracy rate of the first 10
generations of training has increased significantly,and then as the number of
iterations increases,the accuracy rate does not change much.This means that if
there is no new data,the accuracy rate will remain at a relatively stable
level.

⑵From the perspective of the model
itself,the batch_norm layer plays an optimizing role in the model.

From the table2

⑴The choice of lr is not the bigger the
better,nor the smaller the better,but to choose the appropriate value,1e-4 is
the best effect of the model.

⑵The size of batch_size has no effect on the
loss value.It only allows the batch to be processed and the processing time is
less.

⑶Under this model,the Adam optimizer
performs better than SGD optimizer.

Other aspects

⑴In the classification model,it is better to
use the cross-entropy loss function for loss.

⑵under the same model weight,the accuracy of
the prediction model is not static.As the number of tests increases,the
accuracy will increase slightly.

⑶In the process of training the model,loss
does not show a decreasing trend,but a slight shock,and the overall loss is in
a state of decline.
