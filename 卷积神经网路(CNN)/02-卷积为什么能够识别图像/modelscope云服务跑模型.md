# ModeScope（魔搭网）

> 免费的开源网站

https://modelscope.cn/



进入`我的Notebook`

![image-20250411161616445](./assets/image-20250411161616445.png)

进入CPU环境中：

![image-20250411161820123](./assets/image-20250411161820123.png)

启动后，点击`查看NoteBook`

<img src="./assets/image-20250411161958163.png" alt="image-20250411161958163" style="zoom:50%;" />

进入该环境时，为`jupyter 环境`

<img src="./assets/image-20250411162236742.png" alt="image-20250411162236742" style="zoom:50%;" />

上传好文件之后，进入Linux操作控制台

<img src="./assets/image-20250411162344137.png" alt="image-20250411162344137" style="zoom:50%;" />



在控制台中解压刚才上传的文件

<img src="./assets/image-20250411162515269.png" alt="image-20250411162515269" style="zoom:50%;" />

在左边的列表中，右键选择`New Notebook`

<img src="./assets/image-20250411162641684.png" alt="image-20250411162641684" style="zoom:50%;" />

选择需要运行的环境，此处默认选择select

![image-20250411162811358](./assets/image-20250411162811358.png)

上述选择环境后，可以测试一下环境

<img src="./assets/image-20250411162926274.png" alt="image-20250411162926274" style="zoom:50%;" />

如果当前环境缺少需要的库，则可以直接在代码行中输入（在代码中可以用任何pip命令）：

<img src="./assets/image-20250411163047126.png" alt="image-20250411163047126" style="zoom:50%;" />



我们将torch_utils中的代码（因为主代码中引用了关于train_utils的内容），先放入Notebook中，

执行该代码（选择**运行键**）

![image-20250411164744919](./assets/image-20250411164744919.png)

上述代码执行完成后，添加新的代码块

<img src="./assets/image-20250411164918846.png" alt="image-20250411164918846" style="zoom:50%;" />

以下就是新的代码块

<img src="./assets/image-20250411164953338.png" alt="image-20250411164953338" style="zoom:50%;" />

将主代码(数据识别.py)文件中代码，放入新的代码块

![image-20250411165107114](./assets/image-20250411165107114.png)

上述代码中需要删除已经执行过的train_utils的引入

![image-20250411165151985](./assets/image-20250411165151985.png)

修改当前的加载目录为新的目录

![image-20250411165225960](./assets/image-20250411165225960.png)

然后将主方法的代码，删除，并在新的代码块中粘贴

![image-20250411165337957](./assets/image-20250411165337957.png)

![image-20250411165410719](./assets/image-20250411165410719.png)

先运行前面的代码内容

![image-20250411165445669](./assets/image-20250411165445669.png)

为了保存模型，需要新建`save`目录在左侧

<img src="./assets/image-20250411165531283.png" alt="image-20250411165531283" style="zoom:50%;" />

选择运行最后的主代码

![image-20250411165632985](./assets/image-20250411165632985.png)

可以在右上角查看运行占用情况。

<img src="./assets/image-20250411165741549.png" alt="image-20250411165741549" style="zoom:50%;" />

运行完成后，去将save下保存的模型，取出（右键下载它，到本地）