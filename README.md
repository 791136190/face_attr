#### 介绍
多属性训练示例

1、远程显示export DISPLAY=localhost:10.0，xming开启

2、如果cv2和ros的冲突了：
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

3、打开远程的tensorboard
本地cmd进入ssh -L 6006:127.0.0.1:6006 hu@192.168.137.3，激活pytorch环境，然后目录下 tensorboard --logdir runs/train --port 6006，
本地浏览器：http://localhost:6006 或者 http://127.0.0.1:6006 注意浏览器是否显示
使用secureCRT类似软件的话可以使用端口转发功能

4、无法打开文件
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))


5、数据集
afad -> age gender (get)
cacd2000 -> age (get)
celeba -> score galss gender smile hat land (get)
idcard -> gender age (get)
imdb -> gender age (get)
scrfd -> score land (get)
utkface -> age gender (get)
mrfd -> mask

6、统计信息
       score      gender         age         lex         ley         rex         rey       nosex       nosey         lmx         lmy         rmx         rmy       glass       smile         hat        mask
 2.04063e+06 1.08997e+06  1.0546e+06      267272      267272      267272      267272      267272      267272      267272      267272      267272      267272      201409      201409      201409           0
           1       0.534       0.517       0.131       0.131       0.131       0.131       0.131       0.131       0.131       0.131       0.131       0.131      0.0987      0.0987      0.0987           0
        34
      1209       655       340       317       287       303       401       734       816      1047
      1289      1598      2043      2735      4414      6589      7929     13279     23266     29134
     32956     36641     37114     41405     41383     36780     36027     33621     34795     34665
     34674     32835     29898     27674     27919     28518     26382     25763     24455     20157
     18554     19016     18931     18408     18948     18212     16159     16110     15156     14217
     13377     12408     12314     11388     11151      9472      7361      6659      5749      5818
      5130      4436      3571      3180      3362      2837      2436      2122      1748      1925
      1464      1447      1102       998       974       853       752       719       570       567
       475       385       268       295       347       197       153       173       128       162
        77        48        34        29        22        36        13         5        18        49
max age num: 24 41405.0
         5, 
         5,          5,          5,          5,          5,          5,          5,          5,          5,          5, 
         5,          5,          5,          5,          5,          5,          5,        3.1,        1.8,        1.4, 
       1.3,        1.1,        1.1,          1,          1,        1.1,        1.1,        1.2,        1.2,        1.2, 
       1.2,        1.3,        1.4,        1.5,        1.5,        1.5,        1.6,        1.6,        1.7,        2.1, 
       2.2,        2.2,        2.2,        2.2,        2.2,        2.3,        2.6,        2.6,        2.7,        2.9, 
       3.1,        3.3,        3.4,        3.6,        3.7,        4.4,          5,          5,          5,          5, 
         5,          5,          5,          5,          5,          5,          5,          5,          5,          5, 
         5,          5,          5,          5,          5,          5,          5,          5,          5,          5, 
         5,          5,          5,          5,          5,          5,          5,          5,          5,          5, 
         5,          5,          5,          5,          5,          5,          5,          5,          5,          5, 

end check process!!!