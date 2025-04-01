# SL_ParsingAPI

얼굴, 번호판 Segmentation 모듈 & API

위 저장소에서 model 폴더 및 내용물을 다운로드하여 프로젝트 경로로 이동 후

python V3.py

위와 같은 요령으로 실행


# conda env

conda create -n sl-parsing-api python = 3.7

# requirements

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia  
pip install flask  
pip install matplotlib  
pip install pillow
