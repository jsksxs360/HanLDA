# HanLDA

**该项目已合并至 [AHANLP](https://github.com/jsksxs360/AHANLP)，从 2021-05-13 起停止维护。**

[LDA4j](https://github.com/hankcs/LDA4j) 是由 [Hankcs](http://www.hankcs.com/) 编写的一个 LDA(Latent Dirichlet Allocation) 的 java 实现版本，其中的 LdaGibbsSampler.java 直接照搬 Gregor Heinrich 的实现，代码非常精炼（国内 yangliuy 和 ansj 的实现也都是从 Gregor Heinrich 派生出来的）。

HanLDA 是对 LDA4j 的进一步包装，实现了 LDA 模型的存储和读取，并且将使用操作简化为两个过程：

- 训练：在训练语料上进行 LDA 模型的训练，并且生成模型文件。
- 预测：使用训练好的模型文件，推测新文档的主题分布。

### 下载

[HanLDA.jar](https://github.com/jsksxs360/HanLDA/releases/)

## 如何使用

```java
//训练语料，产生 LDA 模型
HanLDA.train("data/mini", 10, "model/lda.model", true);
//预测指定文档的主题分布
System.out.println("---军事_510.txt 的主题分布预测---");
HanLDA.inference("model/lda.model", "data/mini/军事_510.txt", true);
```

训练过程输出的 LDA 模型：

```java
topic 0 :
中国=0.008859160573618033
文化=0.0077097799287296626
历史=0.003943465535727256
社会=0.0036133467402586555
没有=0.0032509913800735997
国家=0.003219278878574916
世界=0.0027026865845008086
政治=0.0026654179755018345
认为=0.0022180062246475803
问题=0.002129156122482994

topic 1 :
美国=0.007659549572894995
训练=0.004257194995556462
系统=0.004084136910473438
日本=0.003963269040941923
部队=0.0038668118598646196
飞机=0.003861138047515428
进行=0.003722444607265373
作战=0.003593516963555568
装备=0.0034978720698204306
演习=0.003439346741040618

topic 2 :
旅游=0.014335208713663285
游客=0.004504258429476157
城市=0.0036694358090017527
旅行社=0.0024064911699769273
成都=0.0021095069963857197
北京=0.002021117788137341
航班=0.002014418941645558
世界=0.0017460205001742246
旅客=0.0016078110818662052
景区=0.001521244253480048

topic 3 :
市场=0.008830707720188975
中国=0.008704106196810836
公司=0.008430801251778176
企业=0.00599257534261676
发展=0.005627063245390042
目前=0.004381558348249937
产品=0.004036993147050587
已经=0.0036887735435382186
服务=0.0035689144563519136
表示=0.0032170971559895255

topic 4 :
比赛=0.00869949608450365
队员=0.0033491917975663043
联赛=0.003037578054930024
球队=0.002598542219298376
冠军=0.0022274409470748636
俱乐部=0.002179714675741507
球员=0.0020511954006965185
决赛=0.002029213860809676
中国=0.0019118309390075863
赛季=0.0018833625470811941

topic 5 :
The=0.0021632935264217566
意思=0.0012299724940318475
It=0.0011399576148711785
理解=0.0011317043748475806
What=9.58498063457519E-4
They=9.366839273818395E-4
In=8.167058712074362E-4
听力=7.860825341462817E-4
译文=7.589398689438345E-4
阅读=7.518802028966632E-4

topic 6 :
公司=0.01922884107340073
股东=0.009080504571550383
股份=0.007691803439552137
搜狐=0.006377377030396266
有限公司=0.00617199586214339
直播员=0.005322478092804105
股权=0.005240278059552533
项目=0.004950165252043496
发行=0.004402764534204148
改革=0.004316040210811883

topic 7 :
没有=0.006507762831397133
生活=0.003912729109722978
孩子=0.0038973300302978424
时间=0.003202574446742164
男人=0.00317566999828129
工作=0.0030584529799752075
知道=0.003056013028050713
不能=0.002832152624721846
可能=0.0027248851775419918
问题=0.0026993631917832306

topic 8 :
即将=0.001691418075344947
太阳=0.0016289197510844353
更名=0.0015991364913806518
合并=0.0015825568432783707
升格=0.0014136756559420714
主编=0.0011934259588234426
专科学校=0.001154736618792733
组建=0.0010912634903498218
赵孟頫=8.99200613634595E-4
概论=8.744079951688637E-4

topic 9 :
工作=0.009879503078775234
专业=0.0075058392075079885
学生=0.006667950831706714
学校=0.005382744404301158
教育=0.005077697074505933
大学=0.004605624209014285
考生=0.00441813418933774
考试=0.004272190854739569
人才=0.00387903396863121
职业=0.0037564588548662328
```

对 **军事_510.txt** 文档的主题分布预测：

```java
---军事_510.txt 的主题分布预测---
topic0-----0.01282051282051282
topic1-----0.8012820512820513
topic2-----0.019230769230769232
topic3-----0.03205128205128205
topic4-----0.01282051282051282
topic5-----0.019230769230769232
topic6-----0.019230769230769232
topic7-----0.01282051282051282
topic8-----0.02564102564102564
topic9-----0.04487179487179487
```

### 说明

- 训练语料来自搜狗分类语料库，存放在 *data/mini* 目录下，已经使用 [HanLP](https://github.com/hankcs/HanLP) 工具分好词，词与词之间使用空格分隔。如果你需要使用自定义语料，也需要分好词。
- 训练时需要指定模型存放路径，训练后除了产生 LDA 模型文件外，还会产生以 `模型文件名.txt` 命名的模型展示文件。内容为模型内容（每个主题下的词分布），每个主题展示出现概率最高的前 20 个词。

## 参考

- [LDA入门与Java实现](http://www.hankcs.com/nlp/lda-java-introduction-and-implementation.html)
