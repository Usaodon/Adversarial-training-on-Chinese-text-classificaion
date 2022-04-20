# Adversarial-training-on-Chinese-text-classificaion

## 项目介绍
基于中文文本分类模型的对抗训练实验。Baseline为TextCNN，对抗实验方法包括FGSM, FGM, PGD以及Free。  
Baseline模型github地址：[中文文本分类模型](https://github.com/649453932/Chinese-Text-Classification-Pytorch)  
## 环境配置  
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX  
## 对抗训练方式
### [FGSM](https://arxiv.org/pdf/1412.6572.pdf)  
FGSM是Goodfellow在2015年提出的方法。模型的梯度可以表示为：  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://latex.codecogs.com/svg.image?\LARGE&space;g&space;=&space;\bigtriangledown&space;_{x}L(\Theta&space;,x,y))  
根据梯度增大的方向，可以使用符号函数：  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://latex.codecogs.com/svg.image?\LARGE&space;r_{adv}=\epsilon&space;\cdot&space;sign(g))  
该方式在模型中的具体应用应为：  
```
          outputs = model(trains)
          model.zero_grad()
          loss = F.cross_entropy(outputs, labels)
          loss.backward()                                         #模型首先前后向一次
          adv_model.attack()                                      #加入扰动
          adv_outputs = model(trains)
          adv_loss = F.cross_entropy(adv_outputs, labels)
          adv_loss.backward()                                     #再一次前后向，累加梯度
          adv_model.restore()                                     #恢复embedding参数
          optimizer.step()                                        #按原来的embedding参数进行更新
```  
### [FGM](https://arxiv.org/pdf/1605.07725.pdf)  
FGM中扰动的取值只有+-eps两个值，很难知晓扰动过大还是过小。  
FGM是对FGSM进行的改进，在扰动上进行了缩放。实验表明，相比FGSM能够获得更好的对抗样本。具体步骤与FGSM类似。  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://latex.codecogs.com/svg.image?\LARGE&space;r_{adv}=\epsilon&space;\tfrac{g}{\left\|&space;g\right\|_{2}})
### [PGD](https://arxiv.org/pdf/1706.06083.pdf)  
PGD论文指出, FGM其实是为使鞍点公式的内部最大化的简单的一步方案。  
一个更具对抗性的示例是经过多步的变体，它本质上是负损失函数上的投影梯度下降。  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://latex.codecogs.com/svg.image?\LARGE&space;r_{adv,t&plus;1}=r_{adv,t}&plus;\alpha&space;\cdot&space;\frac{g}{\left\|&space;g\right\|_{2}}&space;\quad&space;and&space;\quad&space;&space;r_{adv}\in&space;[-\epsilon&space;,\epsilon&space;])  
该方式在模型中的具体应用应为：
```
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()                                       #进行一次前后向，得到梯度
            K = 3
            adv_model.backup_grad()                               #备份此梯度
            for t in range(K):
                adv_model.attack(is_first_attack=(t == 0))        #添加扰动，备份embedding参数
                if t != K - 1:
                    optimizer.zero_grad()                         #如果不是最后一步，则不计算梯度
                else:
                    adv_model.restore_grad()                      #如果是最后一步，载入之前备份的梯度
                adv_outputs = model(trains)
                adv_loss = F.cross_entropy(adv_outputs, labels)
                adv_loss.backward()                               #进行一次前后向，并累加梯度
            adv_model.restore()                                   #恢复embedding参数

            optimizer.step()                                      #在此基础上进行参数更新
```  
### [Free](https://proceedings.neurips.cc/paper/2019/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf)  
FGSM，FGM以及PGD都为模型带来了对抗示例，虽然有效，但是计算量在增加。  
例如PGD中会对同一批样本计算K+1次前后向，然后再更新参数。  
为了提升训练速度，论文提出Free，主要思想是在PGD的基础上，每经过一步就更新一次参数，并复用上一步的扰动。  
训练epoch上限也会变为原来的1/K。  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![](https://latex.codecogs.com/svg.image?\LARGE&space;r_{adv,t&plus;1}=r_{adv,t}&plus;\epsilon&space;\cdot&space;sign(g))  
计算公式类似FGSM，实质上就是重复执行K次的FGSM。  
该方式在模型中的具体应用应为：
```
                optimizer.zero_grad()
                adv_model.attack(r_at)                            #添加扰动
                r_at = r_at.detach()
                r_at.requires_grad_()

                adv_outputs = model(trains)                       
                adv_loss = F.cross_entropy(adv_outputs, labels)

                adv_loss.backward()                               #进行一次前后向

                grad = model.embedding.weight.grad                #记录梯度
                adv_model.restore()                               #恢复embedding参数
                optimizer.step()                                  #在此基础上进行参数更新

                if grad is not None:                              #利用记录的梯度更新扰动
                    r_at = r_at.detach() + epsilon * torch.sign(grad.detach())
                    r_at = torch.clamp(r_at, -epsilon, epsilon)
```  

## 数据集
从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，文本长度在20到30之间。一共10个类别，每类2万条。  
以字为单位输入模型，使用了预训练词向量：[搜狗新闻 Word+Character 300d](https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ)。  
类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。  
Baseline模型为TextCNN。对抗训练方式包括FGSM,FGM, PGD和Free。  
使用precision, recall以及f1-score对训练结果进行评估。  
## 实验结果  
  
|        | precision | recall | f1-score |
|  ----  | ----  | ---- | ---- |
| Baseline  | 0.9146 | 0.9148 | 0.9146 |
| FGSM  | 0.9160 | 0.9156 | 0.9156 |
| FGM  | 0.9195 | 0.9189 | 0.9189 |
| PGD  | 0.9179 | 0.9175 | 0.9174 |
| Free  | 0.9216 | 0.9217 | 0.9216 |  
## 使用方法
```
#训练并测试
#Baseline
python run.py --model TextCNN --adv_mode Baseline

#FGSM
python run.py --model TextCNN --adv_mode FGSM

#FGM
python run.py --model TextCNN --adv_mode FGM

#PGD
python run.py --model TextCNN --adv_mode PGD

#Free
python run.py --model TextCNN --adv_mode Free
```  
## References  
[1]Miyato, Takeru, Andrew M. Dai, and Ian Goodfellow. "Adversarial training methods for semi-supervised text classification." arXiv preprint arXiv:1605.07725 (2016).  
[2]Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).  
[3]Madry, Aleksander, et al. "Towards deep learning models resistant to adversarial attacks." arXiv preprint arXiv:1706.06083 (2017).  
[4]Shafahi, Ali, et al. "Adversarial training for free!." Advances in Neural Information Processing Systems 32 (2019).  
[5]Zhu, Chen, et al. "Freelb: Enhanced adversarial training for language understanding." (2019).  
