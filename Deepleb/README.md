# Deepleb

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

### Downsampling & Upsampling

Donsampling은 주 목적이 차원을 줄여서 적은 메모리로 깊은 Convolution을 할 수 있게 하는 것이다. 보통 stride를 2이상으로 하는 Convolution을 사용하거나, pooling을 사용한다. 이 과정을 진행하면
어쩔 수 없이 위치의 정보를 잃게된다. 마지막에 Fully-Connected Layer를 넣지 않고, Fully Connected Network를 주로 사용하고 FCN 이후 대부분의 모델들에서 사용한다.

Upsampling은 Downsampling을 통해서 받은 결과의 차원을 늘려서 인풋과 같은 차원으로 만들어주는 과정이다. 주로 Strided Transpose Convolution을 사용한다.

## Deepleb v3+

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F97ZQt%2FbtqBgeIucmZ%2FsSBqU5UIhsF7sSJg3D2KfK%2Fimg.png)   

위 이미지를 보면 파란박스인 Encoder와 빨간박스인 Decoder가 나온다.

### Encoder & Decoder

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbs8OkP%2FbtqBdT6OoP7%2F85kD5ZFk5OsrsRlcNvsan1%2Fimg.png)   
U-net의 구조 예시   

빨간선을 기준으로 왼쪽이 encoder 오른쪽이 decoder로 encoder에서는 input이미지를 downsampling해 특징을 추출해 내는 역할을 하고, decoder는 Upsampling을 하며
segmentation map을 만들어 낸다.

U-net과 마찬가지로 deeplab도 encoder가 input image를 downsampling해 decoder 부분에 전달해 주는 것을 볼 수 있다.

### Atrous Convolution
![`이미지`](https://miro.medium.com/max/1130/1*-r7CL0AkeO72MIDpjRxfog.png)   
atrous convolution 예시

Atrous convolution이란 기존 convolution과 다르게 필터 사이에 간격을 두는 것이다. rate라는 값에 따라 그 간격이 결정되고 일반 convolution 필터처럼 간격이 없는 것은 rate = 1에 해당된다.

atrous convolution의 장점은 파라미터의 수를 늘리지 않고도 receptive field를 넓일 수 있다는 것이다.(한 픽셀이 볼 수 있는 영역이 커진다.)
![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbxZP4B%2FbtqBgEAb2HS%2FNdLMGUmN5xPkWQeGkForLK%2Fimg.png)   
위 이미지를 보면 feature map의 해상도가 확실히 차이 나는게 보인다.

### Depthwise Convolution

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FZiVRI%2FbtqBeUcU6RY%2FgRZSJVpahvMsA4SkUVw0KK%2Fimg.png)   
depthwise separable Convolution의 예시

Depthwise seperable convolution이란 한 레이어의 input 채널을 따로 따로 convolution 하고 concatenate 한 후 채널 수를 조정하는 방식이다.
설명을 더 해보면 Depthwise seperable convolution은 크게 Depthwise Convolution과 Pointwise Convolution 부분으로 나눌 수 있다.

Depthwise Convolution은 한 layer의 input channel을 따로 따로 convolution하고 concatenate 하는 것을 말한다.
Pointwise Convolution은 1x1 convolution을 통해 output channel 수를 조정하는 것을 말한다.

이 둘을 진행하고 나면 커널의 크기를 KxK, Output channel 수를 M이라고 할 때, 기존 파라미터 수가 KxKxM 일 때 Depthwise seperable Convolution은
KxK+M이 되므로 파라미터수를 줄일 수 있다.(연산량 감소)

연산량이 감소하였지만 기존 convolution과 성능이 비슷하기에 빨라졌다 할 수 있다.

### Atrous Spatial Pyramid Pooling (ASPP)

![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcBRTL2%2FbtqBikIQVIv%2FZ7mAtovDoETDhgmN82Kwvk%2Fimg.png)   
ASPP의 예시

ASPP는 특정 layer에 대해서 다양한 rate의 Atrous Convolution을 적용해 결과를 concatenate하고 1x1 Convolution한 방법이다.
ASPP를 사용하면 Atrous Convolution rate에 따른 다양한 크기의 물체를 잘 인식해 결과를 내게 되는 장점이 있다.

ASPP는 v2로 넘어갈때 추가되었다.

### Residual learning
![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbtLGfz%2FbtqBika4Ptr%2FQtNtJ6PCEB0282O9ws4511%2Fimg.png)   
Resnet의 residual block 예시

많은 모델에서 활용되고 있는 residual learning이다.

사진을 해석하면 F(x)라는 모델에 input X가 들어가 일련의 과정을 거치면서 자신(identity)인 X가 더해져서 output이 F(x)+X가 나오는 구조이다.
vanishing/exploding gradient 문제를 줄여준다는 장점이 있다.

---

### model structure
![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbeQy5l%2FbtqBpQHRp11%2Fjyrxtr2iTjDFL92mXkbz31%2Fimg.png) ![`이미지`](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbbIG1O%2FbtqBhDWuaa2%2FQlVDt0w7ZDfKFHFFrYMBTK%2Fimg.png)   
deeplab v3+ 모델

오른쪽 그림은 모델의 전체적인 구조를 보여주고 연두색 박스에 해당되는 부분을 보여준다.
왼쪽 그림을 보면 Encoder와 Decoder 부분으로 나뉘는 것을 볼 수 있다.

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F0CzOE%2FbtqBoxbd5Bm%2FR2Av9MkG6xHCy7b35Guf00%2Fimg.png) ![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcfK4CW%2FbtqBoPWV62t%2FSEoWCkiLE1YHpGCVlz9Qt1%2Fimg.png)   
deeplab v3+ 모델

여러 layer들을 거친 후 ASPP를 적용한다.
두 사진의 연두 박스 부분이 ASPP부분이다.

ASPP까지 적용했으면 Decoder 부분에서 Upsampling을 진행해 최종 output을 내게 됩니다.