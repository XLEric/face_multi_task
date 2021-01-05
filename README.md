# face_multi_task（人脸属性多任务模型）
pytorch，face multi task，landmarks，age，gender

## 数据集（DataSets）  
采用数据集：IMDB-WIKI  ，其官方地址如下：
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/  
* 该项目使用的数据为WIKI的“faces only”数据。  
* 且进行了一定的加工修改：    
* 1）去掉了一些错误的年龄数据人脸数据；   
* 2）添加了98个人脸关键点数据；  
* 3）更改了数据标注格式，并保存为json文件，每个图片对应一个json标注文件。   
该项目的数据链接地址为:[百度网盘](https://pan.baidu.com/s/1fZIXNbeTznXwTcPe8LbE8w),提取码：vvyj   
将数据压缩文件wiki_crop_face.zip在根目录进行解压。   

## 预训练模型（Pretrain Model）  
该项目的数据链接地址为:[百度网盘](https://pan.baidu.com/s/1K8Iem3mrbu2w9DuC_VKx7g),提取码：yxq8   

## 项目使用方法(Usage)    

1、模型训练，根目录下运行：python train.py   
2、模型前向，在inference.py脚本中设置模型路径后，在根目录下运行：python inference.py    

  ![sample_1](https://github.com/XLEric/face_multi_task/tree/main/samples/sample1.png)  

## 联系方式 （Contact）   
* E-mails: 305141918@qq.com   
