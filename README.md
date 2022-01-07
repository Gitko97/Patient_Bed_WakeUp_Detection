# Patient_Bed_WakeUp_Detection






<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


### Built With

본 서비스 제공을 위해 다양한 파이썬 라이브러리를 사용하였습니다. 
구체적인 라이브러리 및 관련 버전은 requirement.txt 파일을 통해 찾아볼 수 있습니다.

주요 라이브러리와 프레임워크는 다음과 같습니다.

* [OpenCV](https://opencv.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Twilio](https://www.twilio.com/)



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Twillo
  본 서비스의 무단외출의 SMS 서비스를 사용하기 위해서는 [Twilio](https://www.twilio.com/) 에서 가상의 번호를 발급받아야합니다.
![image](https://user-images.githubusercontent.com/41865809/148550294-e119e62c-4cc3-4d7c-abd8-6f6c101507b3.png)



  위 그림과 같이 3개의 정보를 발급받아 사용할 실제 핸드폰 번호를 포함하여 `SMS.py` 의 4개의 property를 업데이트 해야합니다. (사용하지 않아도 무관합니다)</br>
![image](https://user-images.githubusercontent.com/41865809/148550212-2bf21fd5-28d3-49fe-97cc-98c0935f8944.png)

* Libraries
  테스트 환경은 Anaconda의 가상환경 위에서 진행되었습니다.
  따라서 관련 라이브러리의 설치는 `requirement.txt` 를 참조해주세요.
  
  * requirement.txt
  ```sh
  pip install -r requirements.txt
  ```

* Test video
  본 서비스를 이용하기 위해서는 별도의 동영상이 필요합니다.
  저희는 이를 위해 `testVideo` 폴더에 총 3개의 테스트 샘플을 제공하고 있습니다.

</br>

### Installation & Execution

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Please complete the `Perequisites`
2. Clone the repo
   ```sh
   git clone https://github.com/Gitko97/Patient_Bed_WakeUp_Detection.git
   ```
3. Execute program with some arguments. You can get some help by typing below script
   
   ```sh
   python main.py --h
   ```
   
4. 다음 script 실행을 통해 영상에 맞는 `position.txt` 설정이 필요합니다
   
   ```sh
   python main.py --d true --v (video_path)
   ```
   
   
   `position.txt` 파일은 영상의 문 위치와 배게 위치를 저장하는 파일입니다. </br>
   상단 스크립트 실행 후 마우스 클릭으로 하단 어시스트 문구를 따라 '문'과 '배게'의 바운딩 박스를 설정해 주세요.</br>
   ![image](https://user-images.githubusercontent.com/41865809/148550775-cbb7de80-b2b2-4207-8956-88024a5cb361.png)


    
    * 현재 `position.txt` 파일의 좌표는 `test_2.mov` 영상의 좌표값들 입니다.
5. 프로그램을 실행해 주세요.
   
   ```sh
   python main.py --v (video_path)
   ```
  

<!-- LICENSE -->
## License

Distributed under the *. See `LICENSE.txt` for more information.




<!-- CONTACT -->
## Contact

xcvdsf8216@gmail.com

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew

