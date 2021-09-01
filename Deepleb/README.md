# Deepleb V3

Deepleb은 Semantic Segmentation모델 중 하나이다.   
이 모델을 이용해서 구글폰인 Pixel 2와 Pixel 2X의 Portrait Mode를 구현했다고 한다.
![`이미지`](https://4.bp.blogspot.com/-pQ1j2lyMvMw/WeUbl8BfPdI/AAAAAAAACDk/_nR4-zLdzIoaxOHhbb3AHPRSQRwhb8FfQCLcBGAs/s640/girl-with-the-orange-hat-s.jpg)   

## Segmentation

Segmentation : 모든 픽셀의 레이블을 예측   
자세하게는 Semantic Image Segmentation의 목적은 사진에 있는 모든 픽셀을 해당하는 미리 지정된 개수의 class로 분류하는 것이다. 이미지에 있는 모든 픽셀에 대한 예측을 하는 것이기 때문에 dense prediction 이라고도 불린다.
![`이미지`](https://miro.medium.com/max/686/1*pa-PDx8PxNzeFtOecx8t_Q.png)   

### 다양한 Semantic Segmentation 방법들

AlexNet, VGG 등 분류에 자주 쓰이는 깊은 신경망들은 Semantic Segmentation을 하는데 적합하지 않다. 일단 이런 모델은 parameter의 개수와 차원을 줄이는 layer를 가지고있고
보통 Fully Connected Layer에 의해서 위치에 대한 정보를 잃게 된다. 기존 모델을 사용 못하게 되는 상황이 생기게 된 것이다.

이러한 문제의 중간점을 찾기 위해서 보통 Semantic Segmentation 모델들은 보통 Downsampling & Upsampling의 형태를 가지고 있다.